// Kernel GPU do ES-MCCFR do trucoai — porta fiel de rust/mccfr (motor, chave,
// abstração de ações, raiz) e da tabela v2 (open addressing + filtro de admissão).
// Compilado em tempo de execução pelo NVRTC (trucoai/gpu/cuda.py).
//
// Cada thread faz iterações completas (sorteia uma mão e percorre para os dois
// jogadores, amostragem externa). Atualizações "Hogwild" na tabela global.

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
typedef signed char i8;

#define NA 9
#define TARGET 12
#define MAO11_VALUE 3
#define MAO11_RUN 1
#define IRON_VALUE 1
#define COVERED_FROM_ROUND 1
#define RAISE 6
#define ACCEPT 7
#define RUN 8
#define PH_MAO11 0
#define PH_PLAY 1
#define PH_RESPOND 2
#define PH_HAND_OVER 3
#define PH_MATCH_OVER 4
#define SP_NORMAL 0
#define SP_MAO11 1
#define SP_IRON 2
#define EV_DEAL 0
#define EV_PLAY 1
#define EV_RAISE 2
#define EV_ACCEPT 3
#define EV_RUN 4
#define EV_MAO11 5
#define EV_ROUND 6
#define EV_HAND_END 7
#define EV_FOLD 8
#define TIE (-1)
#define MAX_PROBE 64

__device__ __constant__ u8 BUCKET[14] = {0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7};
__device__ __constant__ u8 LADDER[5] = {1, 3, 6, 9, 12};

struct Ev { u8 t; i8 p; i8 v; };

struct __align__(16) St {  // 144 bytes: cópia em blocos de 16 bytes (uint4)
    u8 scores[2];
    u8 dealer, phase, special, vira;
    i8 hands[2][3];
    u8 rp[3][2], rc[3][2], rcov[3][2];
    u8 round_len[3];
    u8 n_rounds;
    i8 results[3];
    u8 n_results, leader, current, stake;
    i8 pending, raiser, last_raiser, resume;
    Ev log[32];
    u8 log_len;
    u8 _pad[2];
};

struct __align__(64) Slot {
    u64 key;
    u32 visits;
    float regret[NA];
    double avg[NA];
};

struct Ctx {
    Slot* table; u64 mask; u64 limit;
    u8* sketch; u64 sketch_mask; int gate;
    u64* count; u64* dropped;
    const double* eq; int o1; int absv;
};

// ------------------------------------------------------------------ RNG
struct Rng { u64 s[4]; };
__device__ void rng_init(Rng* r, u64 seed) {
    u64 z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9E3779B97F4A7C15ULL;
        u64 v = z;
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = v ^ (v >> 31);
    }
}
__device__ __forceinline__ u64 rotl(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
__device__ u64 rng_next(Rng* r) {
    u64* s = r->s;
    u64 result = rotl(s[1] * 5, 7) * 9;
    u64 t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
__device__ double rng_f64(Rng* r) { return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0); }
__device__ u64 rng_below(Rng* r, u64 n) { return __umul64hi(rng_next(r), n); }

// ------------------------------------------------------------------ cartas
__device__ __forceinline__ int strength(int card, int vira) {
    int r = card >> 2;
    return (r == ((vira >> 2) + 1) % 10) ? 10 + (card & 3) : r;
}
__device__ __forceinline__ int bucket_of(int card, int vira) { return BUCKET[strength(card, vira)]; }

__device__ int next_stake(int v) {
    for (int i = 0; i < 4; i++) if (LADDER[i] == v) return LADDER[i + 1];
    return -1;
}

__device__ void hand_outcome(const i8* res, int n, bool* done, int* winner) {
    int w0 = 0, w1 = 0;
    for (int i = 0; i < n; i++) { w0 += res[i] == 0; w1 += res[i] == 1; }
    *done = false; *winner = -1;
    if (w0 >= 2) { *done = true; *winner = 0; return; }
    if (w1 >= 2) { *done = true; *winner = 1; return; }
    if (n < 2) return;
    int first = res[0];
    if (first == TIE) {
        for (int i = 1; i < n; i++) if (res[i] != TIE) { *done = true; *winner = res[i]; return; }
        if (n < 3) return;
        *done = true; *winner = -1; return;
    }
    for (int i = 1; i < n; i++) if (res[i] == TIE) { *done = true; *winner = first; return; }
    if (n < 3) return;
    *done = true; *winner = first;
}

// ------------------------------------------------------------------ estado
__device__ void push(St* s, int t, int p, int v) {
    s->log[s->log_len].t = (u8)t; s->log[s->log_len].p = (i8)p; s->log[s->log_len].v = (i8)v;
    s->log_len++;
}

__device__ void dealt(St* s, int dealer, int s0, int s1, const u8* deck) {
    int mao = 1 - dealer;
    s->scores[0] = s0; s->scores[1] = s1;
    s->dealer = dealer; s->phase = PH_PLAY; s->special = SP_NORMAL; s->vira = deck[6];
    for (int i = 0; i < 3; i++) {
        s->hands[mao][i] = deck[2 * i];
        s->hands[1 - mao][i] = deck[2 * i + 1];
        s->round_len[i] = 0; s->results[i] = 0;
    }
    s->n_rounds = 1; s->n_results = 0; s->leader = mao; s->current = mao; s->stake = 1;
    s->pending = -1; s->raiser = -1; s->last_raiser = -1; s->resume = -1; s->log_len = 0;
    push(s, EV_DEAL, dealer, s->vira);
    bool a0 = s0 == TARGET - 1, a1 = s1 == TARGET - 1;
    if (a0 && a1) { s->special = SP_IRON; s->stake = IRON_VALUE; }
    else if (a0 || a1) { s->special = SP_MAO11; s->phase = PH_MAO11; s->current = a0 ? 0 : 1; }
}

__device__ __forceinline__ bool terminal(const St* s) { return s->phase == PH_HAND_OVER || s->phase == PH_MATCH_OVER; }
__device__ __forceinline__ bool raises_allowed(const St* s) { return s->special == SP_NORMAL; }
__device__ __forceinline__ bool blind(const St* s, int o1) { return s->special == SP_IRON && !o1; }

__device__ int legal(const St* s, u8* out, int o1) {
    int n = 0;
    if (s->phase == PH_MAO11) { out[0] = ACCEPT; out[1] = RUN; return 2; }
    if (s->phase == PH_RESPOND) {
        if (raises_allowed(s) && next_stake(s->pending) >= 0) out[n++] = RAISE;
        out[n++] = ACCEPT; out[n++] = RUN;
        return n;
    }
    if (s->phase == PH_PLAY) {
        int p = s->current;
        for (int i = 0; i < 3; i++) if (s->hands[p][i] >= 0) out[n++] = i;
        if (s->n_results >= COVERED_FROM_ROUND && s->special != SP_IRON)
            for (int i = 0; i < 3; i++) if (s->hands[p][i] >= 0) out[n++] = 3 + i;
        if (raises_allowed(s) && s->last_raiser != (i8)s->current && next_stake(s->stake) >= 0) out[n++] = RAISE;
        if (o1 && s->special == SP_NORMAL && s->n_results == 0) out[n++] = RUN;
    }
    return n;
}

__device__ void end_hand(St* s, int winner, int points) {
    int pts = winner >= 0 ? points : 0;
    s->pending = -1; s->raiser = -1; s->resume = -1;
    push(s, EV_HAND_END, winner, pts);
    if (winner >= 0) {
        s->scores[winner] += pts;
        if (s->scores[winner] >= TARGET) { s->phase = PH_MATCH_OVER; return; }
    }
    s->phase = PH_HAND_OVER;
}

__device__ void resolve_round(St* s) {
    int r = s->n_rounds - 1;
    int pa = s->rp[r][0], ca = s->rc[r][0], cova = s->rcov[r][0];
    int pb = s->rp[r][1], cb = s->rc[r][1], covb = s->rcov[r][1];
    int sa = cova ? -1 : strength(ca, s->vira);
    int sb = covb ? -1 : strength(cb, s->vira);
    int res = sa > sb ? pa : (sb > sa ? pb : TIE);
    s->results[s->n_results++] = res;
    push(s, EV_ROUND, res, s->n_results - 1);
    bool done; int winner;
    hand_outcome(s->results, s->n_results, &done, &winner);
    if (done) { end_hand(s, winner, winner < 0 ? 0 : s->stake); return; }
    int nxt = res == TIE ? s->leader : res;
    s->n_rounds++; s->leader = nxt; s->current = nxt;
}

__device__ void apply_action(St* s, int a) {
    int p = s->current;
    if (s->phase == PH_MAO11) {
        push(s, EV_MAO11, p, a);
        if (a == ACCEPT) { s->stake = MAO11_VALUE; s->phase = PH_PLAY; s->current = s->leader; }
        else end_hand(s, 1 - p, MAO11_RUN);
        return;
    }
    if (s->phase == PH_RESPOND) {
        if (a == ACCEPT) {
            push(s, EV_ACCEPT, p, s->pending);
            s->stake = s->pending; s->last_raiser = s->raiser; s->pending = -1; s->raiser = -1;
            s->current = s->resume; s->resume = -1; s->phase = PH_PLAY;
        } else if (a == RUN) {
            push(s, EV_RUN, p, s->pending);
            end_hand(s, s->raiser, s->stake);
        } else {
            s->stake = s->pending; s->pending = next_stake(s->pending); s->raiser = p;
            push(s, EV_RAISE, p, s->pending); s->current = 1 - p;
        }
        return;
    }
    // PH_PLAY
    if (a == RAISE) {
        s->pending = next_stake(s->stake); s->raiser = p; s->resume = p;
        push(s, EV_RAISE, p, s->pending); s->phase = PH_RESPOND; s->current = 1 - p;
        return;
    }
    if (a == RUN) { push(s, EV_FOLD, p, s->stake); end_hand(s, 1 - p, s->stake); return; }
    int slot = a % 3; int covered = a >= 3;
    int card = s->hands[p][slot];
    s->hands[p][slot] = -1;
    int r = s->n_rounds - 1; int k = s->round_len[r];
    s->rp[r][k] = p; s->rc[r][k] = card; s->rcov[r][k] = covered;
    s->round_len[r]++;
    push(s, EV_PLAY, p, covered ? -1 : card);
    if (s->round_len[r] == 2) resolve_round(s); else s->current = 1 - p;
}

// Posições da mão de `me` da mais fraca para a mais forte (às cegas: por posição).
__device__ int sorted_slots(const St* s, int me, u8* o, int o1) {
    int n = 0;
    for (int i = 0; i < 3; i++) if (s->hands[me][i] >= 0) o[n++] = i;
    if (!blind(s, o1)) {
        for (int i = 1; i < n; i++) {  // inserção estável por (força, posição)
            u8 x = o[i]; int kx = strength(s->hands[me][x], s->vira); int j = i - 1;
            while (j >= 0) {
                int kj = strength(s->hands[me][o[j]], s->vira);
                if (kj > kx || (kj == kx && o[j] > x)) { o[j + 1] = o[j]; j--; } else break;
            }
            o[j + 1] = x;
        }
    }
    return n;
}

__device__ int abstract_legal(const St* s, u8* abs, u8* order, int o1) {
    u8 conc[NA];
    int n = legal(s, conc, o1);
    int k = sorted_slots(s, s->current, order, o1);
    for (int i = 0; i < n; i++) {
        int a = conc[i];
        if (a >= 6) { abs[i] = a; continue; }
        int pos = 0;
        for (int j = 0; j < k; j++) if (order[j] == a % 3) { pos = j; break; }
        abs[i] = (a / 3) * 3 + pos;
    }
    for (int i = 1; i < n; i++) {  // ordena crescente
        u8 x = abs[i]; int j = i - 1;
        while (j >= 0 && abs[j] > x) { abs[j + 1] = abs[j]; j--; }
        abs[j + 1] = x;
    }
    return n;
}
__device__ __forceinline__ int to_concrete(const u8* order, int ab) { return ab >= 6 ? ab : (ab / 3) * 3 + order[ab % 3]; }

// ------------------------------------------------------------------ chave (hash FNV-1a + splitmix)
#define PUT(b) (h = (h ^ (u64)(u8)(b)) * 0x100000001b3ULL)
__device__ __forceinline__ int sb2(int x) { return x < 4 ? 0 : x < 7 ? 1 : x < 9 ? 2 : x - 6; }
// absv: 0 = v1, 1 = v2a (força exata na mão e no histórico), 2 = v2b (exata só na mão).
__device__ u64 key_hash(const St* s, int me, int o1, int absv) {
    u64 h = 0xcbf29ce484222325ULL;
    PUT("nmi"[s->special]);
    if (absv == 0) {
        PUT(s->scores[me] < 9 ? '0' : '1');
        PUT(s->scores[1 - me] < 9 ? '0' : '1');
    } else {
        PUT('0' + sb2(s->scores[me]));
        PUT('0' + sb2(s->scores[1 - me]));
    }
    PUT(':');
    if (blind(s, o1)) {
        PUT('B');
        for (int i = 0; i < 3; i++) PUT(s->hands[me][i] >= 0 ? '1' : '0');
    } else {
        u8 bs[3]; int k = 0;
        for (int i = 0; i < 3; i++) if (s->hands[me][i] >= 0)
            bs[k++] = absv >= 1 ? strength(s->hands[me][i], s->vira) : bucket_of(s->hands[me][i], s->vira);
        for (int i = 1; i < k; i++) { u8 x = bs[i]; int j = i - 1; while (j >= 0 && bs[j] > x) { bs[j + 1] = bs[j]; j--; } bs[j + 1] = x; }
        for (int i = 0; i < k; i++) PUT((absv >= 1 ? 'a' : '0') + bs[i]);
    }
    PUT(':');
    for (int i = 0; i < s->log_len; i++) {
        const Ev e = s->log[i];
        u8 rel = e.p == me ? 'm' : 'o';
        if (e.t == EV_PLAY) {
            PUT(rel);
            PUT(e.v < 0 ? 'x' : (absv == 1 ? 'a' + strength((u8)e.v, s->vira) : '0' + bucket_of((u8)e.v, s->vira)));
        }
        else if (e.t == EV_RAISE || e.t == EV_MAO11) {
            PUT(rel); PUT(e.t == EV_RAISE ? 'R' : 'M');
            int v = (u8)e.v;
            if (v >= 10) PUT('0' + v / 10);
            PUT('0' + v % 10);
        } else if (e.t == EV_ACCEPT) { PUT(rel); PUT('A'); }
        else if (e.t == EV_ROUND) { PUT('|'); if (e.p >= 0) PUT(e.p == me ? 'w' : 'l'); }
    }
    PUT(':');
    PUT("mpr"[s->phase]);
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h ? h : 1;
}

// ------------------------------------------------------------------ raiz (= sample_root)
__device__ void sample_root(Rng* r, int* dealer, int* s0, int* s1, u8* d7) {
    *dealer = (int)rng_below(r, 2);
    double x = rng_f64(r);
    u64 a, b;
    if (x < 0.5) { a = rng_below(r, TARGET); b = rng_below(r, TARGET); }
    else if (x < 0.85) { a = rng_below(r, 4); b = rng_below(r, 4); }
    else {
        a = TARGET - 1; b = rng_below(r, TARGET);
        if (rng_f64(r) < 0.5) { u64 t = a; a = b; b = t; }
    }
    u8 deck[40];
    for (int i = 0; i < 40; i++) deck[i] = i;
    for (int i = 0; i < 7; i++) {
        int j = i + (int)rng_below(r, 40 - i);
        u8 t = deck[i]; deck[i] = deck[j]; deck[j] = t;
    }
    for (int i = 0; i < 7; i++) d7[i] = deck[i];
    *s0 = (int)a; *s1 = (int)b;
}

// ------------------------------------------------------------------ tabela
__device__ long long find_slot(const Ctx* c, u64 h) {
    u64 i = h & c->mask;
    for (int p = 0; p < MAX_PROBE; p++) {
        Slot* s = &c->table[i];
        u64 k = *((volatile u64*)&s->key);
        if (k == h) return (long long)i;
        if (k == 0) {
            if (c->gate > 1) {
                u8* cnt = &c->sketch[(h >> 17) & c->sketch_mask];
                u8 v = *((volatile u8*)cnt);
                if (v < c->gate - 1) { *cnt = v + 1; return -1; }
            }
            if (*((volatile u64*)c->count) >= c->limit) { atomicAdd(c->dropped, 1ULL); return -1; }
            u64 old = atomicCAS(&s->key, 0ULL, h);
            if (old == 0) { atomicAdd(c->count, 1ULL); return (long long)i; }
            if (old == h) return (long long)i;
        }
        i = (i + 1) & c->mask;
    }
    atomicAdd(c->dropped, 1ULL);
    return -1;
}

__device__ __forceinline__ double eq_get(const Ctx* c, int me, int opp) {
    if (me >= TARGET) return 1.0;
    if (opp >= TARGET) return 0.0;
    return c->eq[me * 12 + opp];
}

__device__ int weighted(Rng* r, const double* w, int n) {
    double tot = 0; for (int i = 0; i < n; i++) tot += w[i];
    double u = rng_f64(r) * tot;
    for (int i = 0; i < n; i++) { if (u < w[i]) return i; u -= w[i]; }
    return n - 1;
}

__device__ double traverse(const Ctx* c, St* s, int trav, double before, double t, Rng* rng) {
    if (terminal(s)) return eq_get(c, s->scores[trav], s->scores[1 - trav]) - before;
    int p = s->current;
    long long idx = find_slot(c, key_hash(s, p, c->o1, c->absv));
    u8 abs[NA], order[3];
    int n = abstract_legal(s, abs, order, c->o1);
    double sigma[NA];
    double tot = 0;
    Slot* sl = idx >= 0 ? &c->table[idx] : 0;
    if (sl) {
        sl->visits = sl->visits + 1 < sl->visits ? sl->visits : sl->visits + 1;
        for (int k = 0; k < n; k++) { double r = fmax((double)sl->regret[abs[k]], 0.0); sigma[k] = r; tot += r; }
    }
    if (tot > 0) { for (int k = 0; k < n; k++) sigma[k] /= tot; }
    else { for (int k = 0; k < n; k++) sigma[k] = 1.0 / n; }
    if (p != trav) {
        if (sl) for (int k = 0; k < n; k++) sl->avg[abs[k]] += t * sigma[k];
        int k = weighted(rng, sigma, n);
        apply_action(s, to_concrete(order, abs[k]));
        return traverse(c, s, trav, before, t, rng);
    }
    double vals[NA]; double v = 0;
    for (int k = 0; k < n; k++) {
        St child = *s;
        apply_action(&child, to_concrete(order, abs[k]));
        vals[k] = traverse(c, &child, trav, before, t, rng);
        v += sigma[k] * vals[k];
    }
    if (sl) for (int k = 0; k < n; k++) sl->regret[abs[k]] = (float)fmax((double)sl->regret[abs[k]] + vals[k] - v, 0.0);
    return v;
}

extern "C" __global__ void mccfr_train(Slot* table, u64 mask, u64 limit, u8* sketch, u64 sketch_mask,
                                       int gate, u64* count, u64* dropped, const double* eq, int o1,
                                       u64 seed, u64 iter_base, int iters_per_thread, int absv) {
    Ctx c; c.absv = absv; c.table = table; c.mask = mask; c.limit = limit; c.sketch = sketch; c.sketch_mask = sketch_mask;
    c.gate = gate; c.count = count; c.dropped = dropped; c.eq = eq; c.o1 = o1;
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 nthreads = (u64)gridDim.x * blockDim.x;
    Rng rng; rng_init(&rng, seed * 0x9E3779B97F4A7C15ULL + iter_base * 0xD1B54A32D192ED03ULL + tid);
    for (int k = 0; k < iters_per_thread; k++) {
        double t = (double)(iter_base + (u64)k * nthreads + tid + 1);
        int dealer, s0, s1; u8 d7[7];
        sample_root(&rng, &dealer, &s0, &s1, d7);
        St root; dealt(&root, dealer, s0, s1, d7);
        int sc[2] = {s0, s1};
        for (int trav = 0; trav < 2; trav++) {
            double before = eq_get(&c, sc[trav], sc[1 - trav]);
            St s = root;
            traverse(&c, &s, trav, before, t, &rng);
        }
    }
}

// Autoteste: cada thread joga uma mão com ações aleatórias e grava, por decisão,
// (jogador, hash da chave, ações abstratas legais em bitmask, ação concreta).
// Cabeçalho por jogo: dealer, placar, 7 cartas. Validado contra o motor Python.
#define ST_MAX 48
extern "C" __global__ void selftest(u64 seed, int o1, int games, int* head, u64* recs, int* nrec, int absv) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= games) return;
    Rng rng; rng_init(&rng, seed * 1000003ULL + g);
    int dealer, s0, s1; u8 d7[7];
    sample_root(&rng, &dealer, &s0, &s1, d7);
    St s; dealt(&s, dealer, s0, s1, d7);
    int* hh = head + g * 12;
    hh[0] = dealer; hh[1] = s0; hh[2] = s1;
    for (int i = 0; i < 7; i++) hh[3 + i] = d7[i];
    int n = 0;
    while (!terminal(&s) && n < ST_MAX) {
        u8 abs[NA], order[3];
        int k = abstract_legal(&s, abs, order, o1);
        int mask = 0; for (int i = 0; i < k; i++) mask |= 1 << abs[i];
        int ab = abs[rng_below(&rng, k)];
        int a = to_concrete(order, ab);
        u64* r = recs + ((u64)g * ST_MAX + n) * 2;
        r[0] = key_hash(&s, s.current, o1, absv);
        r[1] = ((u64)s.current << 32) | ((u64)mask << 8) | (u64)a;
        apply_action(&s, a);
        n++;
    }
    nrec[g] = n;
    hh[10] = s.scores[0]; hh[11] = s.scores[1];
}

// =====================================================================
// Travessia em ondas (wavefront): uma thread por NÓ de uma profundidade, de
// milhares de árvores de amostragem externa ao mesmo tempo. Mesma matemática do
// traverse() recursivo: expand (desce, atualiza média nos nós do adversário),
// emit (gera os filhos) e backup (sobe, atualiza arrependimentos RM+).
// =====================================================================

struct NodeMeta {
    long long slot;   // entrada na tabela (-1: sem entrada)
    float sigma[NA];  // estratégia atual (regret matching)
    u8 abs[NA];       // ações abstratas legais (ordenadas)
    u8 order[3];      // posições da mão da mais fraca à mais forte
    u8 n;             // nº de ações
    u8 kind;          // 0 terminal, 1 adversário (1 filho), 2 jogador que treina (n filhos)
    u8 pick;          // ação amostrada (adversário)
    unsigned short explore;  // bits das ações exploradas (quem treina; poda estilo Pluribus)
};

__device__ __forceinline__ u64 mix64(u64 x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; return x ^ (x >> 31);
}

// Raízes: 2 nós por mão sorteada (um por jogador que treina).
extern "C" __global__ void wf_roots(int B, u64 seed, u64 iter_base, St* states, int* root_of,
                                    u8* rtrav, double* rbefore, double* rt, const double* eq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    Rng rng; rng_init(&rng, mix64(seed ^ (iter_base + i) * 0x9E3779B97F4A7C15ULL));
    int dealer, s0, s1; u8 d7[7];
    sample_root(&rng, &dealer, &s0, &s1, d7);
    St s; dealt(&s, dealer, s0, s1, d7);
    int sc[2] = {s0, s1};
    for (int trav = 0; trav < 2; trav++) {
        int j = 2 * i + trav;
        states[j] = s; root_of[j] = j;
        rtrav[j] = trav;
        int me = sc[trav], op = sc[1 - trav];
        rbefore[j] = me >= TARGET ? 1.0 : (op >= TARGET ? 0.0 : eq[me * 12 + op]);
        rt[j] = (double)(iter_base + i + 1);
    }
}

extern "C" __global__ void wf_expand(int N, const St* states, const int* root_of, const u8* rtrav,
                                     const double* rbefore, const double* rt, NodeMeta* meta, int* nchild,
                                     double* val, Slot* table, u64 mask, u64 limit, u8* sketch, u64 sketch_mask,
                                     int gate, u64* count, u64* dropped, const double* eq, int o1, u64 rng_key,
                                     double prune_p, double prune_after, int absv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Ctx c; c.absv = absv; c.table = table; c.mask = mask; c.limit = limit; c.sketch = sketch; c.sketch_mask = sketch_mask;
    c.gate = gate; c.count = count; c.dropped = dropped; c.eq = eq; c.o1 = o1;
    const St* s = &states[i];
    int r = root_of[i];
    int trav = rtrav[r];
    NodeMeta m;
    if (terminal(s)) {
        m.kind = 0; m.n = 0; m.slot = -1;
        val[i] = eq_get(&c, s->scores[trav], s->scores[1 - trav]) - rbefore[r];
        meta[i] = m; nchild[i] = 0;
        return;
    }
    int p = s->current;
    m.slot = find_slot(&c, key_hash(s, p, o1, absv));
    m.n = abstract_legal(s, m.abs, m.order, o1);
    int n = m.n;
    double tot = 0, sg[NA];
    Slot* sl = m.slot >= 0 ? &table[m.slot] : 0;
    if (sl) {
        sl->visits = sl->visits + 1 < sl->visits ? sl->visits : sl->visits + 1;
        for (int k = 0; k < n; k++) { double x = fmax((double)sl->regret[m.abs[k]], 0.0); sg[k] = x; tot += x; }
    }
    for (int k = 0; k < n; k++) sg[k] = tot > 0 ? sg[k] / tot : 1.0 / n;
    for (int k = 0; k < n; k++) m.sigma[k] = (float)sg[k];
    if (p != trav) {
        if (sl) { double t = rt[r]; for (int k = 0; k < n; k++) sl->avg[m.abs[k]] += t * sg[k]; }
        double u = (double)(mix64(rng_key ^ ((u64)i * 0x9E3779B97F4A7C15ULL)) >> 11) * (1.0 / 9007199254740992.0);
        int pick = n - 1;
        for (int k = 0; k < n; k++) { if (u < sg[k]) { pick = k; break; } u -= sg[k]; }
        m.kind = 1; m.pick = pick; nchild[i] = 1;
    } else {
        // Poda: ação com σ=0 (arrependimento RM+ nulo) explorada só com prob. 1-prune_p.
        unsigned short ex = 0; int cnt = 0;
        bool prune = prune_p > 0 && rt[r] > prune_after && tot > 0;
        for (int k = 0; k < n; k++) {
            bool skip = false;
            if (prune && sg[k] == 0.0) {
                double u = (double)(mix64(rng_key ^ ((u64)i * 0x9E3779B97F4A7C15ULL) ^ ((u64)(k + 1) << 56)) >> 11)
                           * (1.0 / 9007199254740992.0);
                skip = u < prune_p;
            }
            if (!skip) { ex |= (unsigned short)(1u << k); cnt++; }
        }
        m.kind = 2; m.explore = ex; nchild[i] = cnt;
    }
    meta[i] = m;
}

// offs: prefixo exclusivo de nchild (posição do 1º filho de cada nó no nível seguinte).
extern "C" __global__ void wf_emit(int N, const St* states, const int* root_of, const NodeMeta* meta,
                                   const int* nchild, const int* offs, St* out_states, int* out_root) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int nc = nchild[i];
    if (nc == 0) return;
    const NodeMeta* m = &meta[i];
    int base = offs[i];
    for (int k = 0; k < nc; k++) {
        int ak = m->kind == 1 ? m->pick : k;
        St s = states[i];
        apply_action(&s, to_concrete(m->order, m->abs[ak]));
        out_states[base + k] = s;
        out_root[base + k] = root_of[i];
    }
}

// Sobe um nível: valor de cada nó a partir dos filhos; RM+ nos nós de quem treina.
extern "C" __global__ void wf_backup(int N, const NodeMeta* meta, const int* nchild, const int* offs,
                                     const double* child_val, double* val, Slot* table) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const NodeMeta* m = &meta[i];
    if (m->kind == 0) return;  // terminal: valor já calculado no expand
    int base = offs[i];
    if (m->kind == 1) { val[i] = child_val[base]; return; }
    double v = 0, cv[NA];
    int c = 0;
    for (int k = 0; k < m->n; k++) {
        if ((m->explore >> k) & 1) { cv[k] = child_val[base + c++]; v += (double)m->sigma[k] * cv[k]; }
    }
    val[i] = v;
    if (m->slot >= 0) {
        Slot* sl = &table[m->slot];
        for (int k = 0; k < m->n; k++) {
            if (!((m->explore >> k) & 1)) continue;  // podada: σ=0, arrependimento inalterado
            float* rg = &sl->regret[m->abs[k]];
            *rg = (float)fmax((double)*rg + cv[k] - v, 0.0);
        }
    }
}

extern "C" __global__ void sizes(int* out) { out[0] = sizeof(St); out[1] = sizeof(NodeMeta); out[2] = sizeof(Slot); }


// Uma thread por FILHO (equilibrado), cópia do estado do pai em blocos de 16 bytes.
extern "C" __global__ void wf_emit2(int M, const St* states, const int* root_of, const NodeMeta* meta,
                                    const int* offs, const long long* parent, St* out_states, int* out_root) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;
    int i = (int)parent[j];
    int k = j - offs[i];
    const NodeMeta* m = &meta[i];
    int ak;
    if (m->kind == 1) ak = m->pick;
    else {  // k-ésima ação explorada
        int c = -1; ak = 0;
        for (int a = 0; a < m->n; a++) if ((m->explore >> a) & 1) { if (++c == k) { ak = a; break; } }
    }
    St s;
    const uint4* src = (const uint4*)&states[i];
    uint4* dst = (uint4*)&s;
#pragma unroll
    for (int w = 0; w < (int)(sizeof(St) / 16); w++) dst[w] = src[w];
    apply_action(&s, to_concrete(m->order, m->abs[ak]));
    uint4* o = (uint4*)&out_states[j];
#pragma unroll
    for (int w = 0; w < (int)(sizeof(St) / 16); w++) o[w] = dst[w];
    out_root[j] = root_of[i];
}
