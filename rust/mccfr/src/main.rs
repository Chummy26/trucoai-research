//! Núcleo em Rust do MCCFR do trucoai.
//!
//! Porta fiel de `trucoai/engine.py` (regulamento `paulista-1v1-h0`, os valores
//! de `DEFAULT_RULES` estão fixos abaixo), de `trucoai/cfr/abstraction.py`
//! (`info_key`) e de `trucoai/cfr/mccfr.py` (ES-MCCFR + RM+ + média linear,
//! utilidade = variação de equidade da partida, mesmo sorteio de placar na raiz).
//!
//! Fronteira de informação: a chave de infoset (`State::key`) lê só as próprias
//! cartas, o placar, o tipo de mão, a fase e o log público (onde cartas
//! encobertas aparecem como -1). `tests/test_rust_core.py` compara, jogada a
//! jogada, a chave, as ações abstratas legais e o resultado com o motor Python
//! e com `info_key(observe(...))`.
//!
//! Diferença em relação ao treino Python: as threads compartilham UMA tabela
//! (CFR assíncrono, estilo Hogwild, com travas por fatia), em vez de somar
//! tabelas independentes no fim.
//!
//! Uso:
//!   mccfr train --eq <tabela.txt> --out <politica.tsv> --summary <resumo.json>
//!               --seed S --threads N --seconds X [--iterations N]
//!               --max-nodes M --save-min-visits V
//!   mccfr selftest --seed S --games N      (jogos aleatórios para o teste Python)

mod v2;

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Regulamento (= DEFAULT_RULES de trucoai/rules.py, versão paulista-1v1-h0)
// ---------------------------------------------------------------------------
const RULES_VERSION: &str = "paulista-1v1-h0";
const RULES_VERSION_O1: &str = "paulista-1v1-o1";
// `--rules o1`: regulamento observado na PlayTruco (trucoai.rules.PLAYTRUCO_O1):
// correr na própria vez só na 1ª rodada de mão normal; mão de ferro com cartas
// visíveis (iron_blind=false). Padrão: h0 (treinos já registrados).
static O1: AtomicBool = AtomicBool::new(false);

#[inline]
fn o1() -> bool {
    O1.load(Ordering::Relaxed)
}

// `--abstraction v1|v2a|v2b` (= trucoai.cfr.abstraction.info_key_v): 0, 1, 2.
static ABSV: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[inline]
fn absv() -> u8 {
    ABSV.load(Ordering::Relaxed)
}

fn abs_name() -> &'static str {
    ["v1", "v2a", "v2b"][absv() as usize]
}

fn rules_version() -> &'static str {
    if o1() { RULES_VERSION_O1 } else { RULES_VERSION }
}
const TARGET: u8 = 12;
const LADDER: [u8; 5] = [1, 3, 6, 9, 12];
const MAO11_VALUE: u8 = 3;
const MAO11_RUN_POINTS: u8 = 1;
const IRON_VALUE: u8 = 1;
const COVERED_FROM_ROUND: u8 = 1;
// cap_raise_at_needed=false, mao11_allows_truco=false, iron_blind=true,
// allow_covered=true, all_tied="nobody", tie_next_leader="same_leader",
// allow_fold_on_turn=false.

const RAISE: u8 = 6;
const ACCEPT: u8 = 7;
const RUN: u8 = 8;
const NUM_ACTIONS: usize = 9;

const PH_MAO11: u8 = 0;
const PH_PLAY: u8 = 1;
const PH_RESPOND: u8 = 2;
const PH_HAND_OVER: u8 = 3;
const PH_MATCH_OVER: u8 = 4;

const SP_NORMAL: u8 = 0;
const SP_MAO11: u8 = 1;
const SP_IRON: u8 = 2;

const EV_DEAL: u8 = 0;
const EV_PLAY: u8 = 1;
const EV_RAISE: u8 = 2;
const EV_ACCEPT: u8 = 3;
const EV_RUN: u8 = 4;
const EV_MAO11: u8 = 5;
const EV_ROUND: u8 = 6;
const EV_HAND_END: u8 = 7;
const EV_FOLD: u8 = 8; // correr na própria vez (O1); não entra na chave

const TIE: i8 = -1;

#[inline]
fn strength(card: u8, vira: u8) -> i8 {
    let r = card >> 2;
    if r == ((vira >> 2) + 1) % 10 {
        10 + (card & 3) as i8
    } else {
        r as i8
    }
}

// Força 0..13 -> faixa 0..7 (igual a abstraction._BUCKET)
const BUCKET: [u8; 14] = [0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7];

#[inline]
fn bucket(card: u8, vira: u8) -> u8 {
    BUCKET[strength(card, vira) as usize]
}

// ---------------------------------------------------------------------------
// RNG (xoshiro256**, semeado por splitmix64)
// ---------------------------------------------------------------------------
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut s = [0u64; 4];
        for x in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut v = z;
            v = (v ^ (v >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94D049BB133111EB);
            *x = v ^ (v >> 31);
        }
        Rng(s)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
    #[inline]
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    #[inline]
    fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }
    fn weighted(&mut self, w: &[f64]) -> usize {
        let tot: f64 = w.iter().sum();
        let mut u = self.f64() * tot;
        for (i, &x) in w.iter().enumerate() {
            if u < x {
                return i;
            }
            u -= x;
        }
        w.len() - 1
    }
}

// ---------------------------------------------------------------------------
// Estado completo da mão (contém informação privada: só o treino o percorre)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct Ev {
    t: u8,
    p: i8,
    v: i8,
}

#[derive(Clone, Copy)]
struct State {
    scores: [u8; 2],
    dealer: u8,
    phase: u8,
    special: u8,
    vira: u8,
    hands: [[i8; 3]; 2],
    rounds: [[(u8, u8, bool); 2]; 3],
    round_len: [u8; 3],
    n_rounds: u8,
    results: [i8; 3],
    n_results: u8,
    leader: u8,
    current: u8,
    stake: u8,
    pending: i8,
    raiser: i8,
    last_raiser: i8,
    resume: i8,
    log: [Ev; 32],
    log_len: u8,
}

fn hand_outcome(results: &[i8]) -> (bool, i8) {
    let w0 = results.iter().filter(|&&r| r == 0).count();
    let w1 = results.iter().filter(|&&r| r == 1).count();
    if w0 >= 2 {
        return (true, 0);
    }
    if w1 >= 2 {
        return (true, 1);
    }
    let n = results.len();
    if n < 2 {
        return (false, -1);
    }
    let first = results[0];
    if first == TIE {
        if let Some(&d) = results[1..].iter().find(|&&r| r != TIE) {
            return (true, d);
        }
        if n < 3 {
            return (false, -1);
        }
        return (true, -1); // all_tied = "nobody"
    }
    if results[1..].contains(&TIE) {
        return (true, first);
    }
    if n < 3 {
        return (false, -1);
    }
    (true, first)
}

fn next_stake(v: u8) -> i8 {
    match LADDER.iter().position(|&x| x == v) {
        Some(i) if i + 1 < LADDER.len() => LADDER[i + 1] as i8,
        _ => -1,
    }
}

impl State {
    /// Equivale a `new_match(first_dealer=dealer)`, placar fixado, e `deal(deck)`.
    fn dealt(dealer: u8, scores: [u8; 2], deck: &[u8; 7]) -> State {
        let mao = 1 - dealer;
        let mut s = State {
            scores,
            dealer,
            phase: PH_PLAY,
            special: SP_NORMAL,
            vira: deck[6],
            hands: [[-1; 3]; 2],
            rounds: [[(0, 0, false); 2]; 3],
            round_len: [0; 3],
            n_rounds: 1,
            results: [0; 3],
            n_results: 0,
            leader: mao,
            current: mao,
            stake: 1,
            pending: -1,
            raiser: -1,
            last_raiser: -1,
            resume: -1,
            log: [Ev { t: 0, p: 0, v: 0 }; 32],
            log_len: 0,
        };
        for i in 0..3 {
            s.hands[mao as usize][i] = deck[2 * i] as i8;
            s.hands[(1 - mao) as usize][i] = deck[2 * i + 1] as i8;
        }
        s.push(EV_DEAL, dealer as i8, s.vira as i8);
        let a11 = [scores[0] == TARGET - 1, scores[1] == TARGET - 1];
        if a11[0] && a11[1] {
            s.special = SP_IRON;
            s.stake = IRON_VALUE;
        } else if a11[0] || a11[1] {
            s.special = SP_MAO11;
            s.phase = PH_MAO11;
            s.current = if a11[0] { 0 } else { 1 };
        }
        s
    }

    #[inline]
    fn push(&mut self, t: u8, p: i8, v: i8) {
        self.log[self.log_len as usize] = Ev { t, p, v };
        self.log_len += 1;
    }

    #[inline]
    fn terminal(&self) -> bool {
        self.phase == PH_HAND_OVER || self.phase == PH_MATCH_OVER
    }

    #[inline]
    fn raises_allowed(&self) -> bool {
        self.special == SP_NORMAL // mão de ferro e mão de onze sem truco
    }

    #[inline]
    fn blind(&self) -> bool {
        self.special == SP_IRON && !o1() // h0: iron_blind = true; O1: false
    }

    /// Ações concretas legais, na mesma ordem de `GameState.legal_actions`.
    fn legal(&self, out: &mut [u8; NUM_ACTIONS]) -> usize {
        let mut n = 0;
        match self.phase {
            PH_MAO11 => {
                out[0] = ACCEPT;
                out[1] = RUN;
                n = 2;
            }
            PH_RESPOND => {
                if self.raises_allowed() && next_stake(self.pending as u8) >= 0 {
                    out[n] = RAISE;
                    n += 1;
                }
                out[n] = ACCEPT;
                out[n + 1] = RUN;
                n += 2;
            }
            PH_PLAY => {
                let p = self.current as usize;
                for i in 0..3 {
                    if self.hands[p][i] >= 0 {
                        out[n] = i as u8;
                        n += 1;
                    }
                }
                if self.n_results >= COVERED_FROM_ROUND && self.special != SP_IRON {
                    for i in 0..3 {
                        if self.hands[p][i] >= 0 {
                            out[n] = 3 + i as u8;
                            n += 1;
                        }
                    }
                }
                if self.raises_allowed()
                    && self.last_raiser != self.current as i8
                    && next_stake(self.stake) >= 0
                {
                    out[n] = RAISE;
                    n += 1;
                }
                // O1: fold_on_turn_max_round = 0, só em mão normal
                if o1() && self.special == SP_NORMAL && self.n_results == 0 {
                    out[n] = RUN;
                    n += 1;
                }
            }
            _ => {}
        }
        n
    }

    fn apply(&mut self, a: u8) {
        let p = self.current;
        match self.phase {
            PH_MAO11 => {
                self.push(EV_MAO11, p as i8, a as i8);
                if a == ACCEPT {
                    self.stake = MAO11_VALUE;
                    self.phase = PH_PLAY;
                    self.current = self.leader;
                } else {
                    self.end_hand(1 - p as i8, MAO11_RUN_POINTS);
                }
            }
            PH_RESPOND => {
                if a == ACCEPT {
                    self.push(EV_ACCEPT, p as i8, self.pending);
                    self.stake = self.pending as u8;
                    self.last_raiser = self.raiser;
                    self.pending = -1;
                    self.raiser = -1;
                    self.current = self.resume as u8;
                    self.resume = -1;
                    self.phase = PH_PLAY;
                } else if a == RUN {
                    self.push(EV_RUN, p as i8, self.pending);
                    let (w, st) = (self.raiser, self.stake);
                    self.end_hand(w, st);
                } else {
                    self.stake = self.pending as u8;
                    self.pending = next_stake(self.pending as u8);
                    self.raiser = p as i8;
                    self.push(EV_RAISE, p as i8, self.pending);
                    self.current = 1 - p;
                }
            }
            PH_PLAY => {
                if a == RAISE {
                    self.pending = next_stake(self.stake);
                    self.raiser = p as i8;
                    self.resume = p as i8;
                    self.push(EV_RAISE, p as i8, self.pending);
                    self.phase = PH_RESPOND;
                    self.current = 1 - p;
                    return;
                }
                if a == RUN {
                    self.push(EV_FOLD, p as i8, self.stake as i8);
                    let st = self.stake;
                    self.end_hand(1 - p as i8, st);
                    return;
                }
                let slot = (a % 3) as usize;
                let covered = a >= 3;
                let card = self.hands[p as usize][slot] as u8;
                self.hands[p as usize][slot] = -1;
                let r = (self.n_rounds - 1) as usize;
                self.rounds[r][self.round_len[r] as usize] = (p, card, covered);
                self.round_len[r] += 1;
                self.push(EV_PLAY, p as i8, if covered { -1 } else { card as i8 });
                if self.round_len[r] == 2 {
                    self.resolve_round();
                } else {
                    self.current = 1 - p;
                }
            }
            _ => panic!("ação em estado terminal"),
        }
    }

    fn resolve_round(&mut self) {
        let r = (self.n_rounds - 1) as usize;
        let [(pa, ca, cova), (pb, cb, covb)] = self.rounds[r];
        let sa = if cova { -1 } else { strength(ca, self.vira) };
        let sb = if covb { -1 } else { strength(cb, self.vira) };
        let res: i8 = if sa > sb {
            pa as i8
        } else if sb > sa {
            pb as i8
        } else {
            TIE
        };
        self.results[self.n_results as usize] = res;
        self.n_results += 1;
        self.push(EV_ROUND, res, (self.n_results - 1) as i8);
        let (done, winner) = hand_outcome(&self.results[..self.n_results as usize]);
        if done {
            if winner < 0 {
                self.end_hand(-1, 0);
            } else {
                let st = self.stake;
                self.end_hand(winner, st);
            }
            return;
        }
        let nxt = if res == TIE { self.leader } else { res as u8 };
        self.n_rounds += 1;
        self.leader = nxt;
        self.current = nxt;
    }

    fn end_hand(&mut self, winner: i8, points: u8) {
        let pts = if winner >= 0 { points } else { 0 };
        self.pending = -1;
        self.raiser = -1;
        self.resume = -1;
        self.push(EV_HAND_END, winner, pts as i8);
        if winner >= 0 {
            self.scores[winner as usize] += pts;
            if self.scores[winner as usize] >= TARGET {
                self.phase = PH_MATCH_OVER;
                return;
            }
        }
        self.phase = PH_HAND_OVER;
    }

    /// Posições da mão de `me`, da carta mais fraca para a mais forte
    /// (`mccfr.sorted_slots`). Às cegas: ordem das posições.
    fn sorted_slots(&self, me: u8) -> ([u8; 3], usize) {
        let h = &self.hands[me as usize];
        let mut o = [0u8; 3];
        let mut n = 0;
        for i in 0..3 {
            if h[i] >= 0 {
                o[n] = i as u8;
                n += 1;
            }
        }
        if !self.blind() {
            let vira = self.vira;
            o[..n].sort_by_key(|&i| (strength(h[i as usize] as u8, vira), i));
        }
        (o, n)
    }

    /// Chave de infoset idêntica a `info_key(observe(state, me))`.
    /// Lê apenas informação legítima de `me`.
    fn key(&self, me: u8, buf: &mut [u8; 128]) -> usize {
        let mut n = 0;
        let mut put = |b: u8, n: &mut usize| {
            buf[*n] = b;
            *n += 1;
        };
        let av = absv();
        let sbk = |s: u8| {
            if av == 0 {
                if s < 9 { b'0' } else { b'1' }
            } else {
                b'0' + if s < 4 { 0 } else if s < 7 { 1 } else if s < 9 { 2 } else { s - 6 }
            }
        };
        put(b"nmi"[self.special as usize], &mut n);
        put(sbk(self.scores[me as usize]), &mut n);
        put(sbk(self.scores[1 - me as usize]), &mut n);
        put(b':', &mut n);
        let h = &self.hands[me as usize];
        if self.blind() {
            put(b'B', &mut n);
            for i in 0..3 {
                put(if h[i] >= 0 { b'1' } else { b'0' }, &mut n);
            }
        } else {
            let mut bs = [0u8; 3];
            let mut k = 0;
            for i in 0..3 {
                if h[i] >= 0 {
                    bs[k] = if av >= 1 { strength(h[i] as u8, self.vira) as u8 } else { bucket(h[i] as u8, self.vira) };
                    k += 1;
                }
            }
            bs[..k].sort_unstable();
            for &b in &bs[..k] {
                put(if av >= 1 { b'a' + b } else { b'0' + b }, &mut n);
            }
        }
        put(b':', &mut n);
        for e in &self.log[..self.log_len as usize] {
            let rel = if e.p == me as i8 { b'm' } else { b'o' };
            match e.t {
                EV_PLAY => {
                    put(rel, &mut n);
                    put(if e.v < 0 {
                        b'x'
                    } else if av == 1 {
                        b'a' + strength(e.v as u8, self.vira) as u8
                    } else {
                        b'0' + bucket(e.v as u8, self.vira)
                    }, &mut n);
                }
                EV_RAISE | EV_MAO11 => {
                    put(rel, &mut n);
                    put(if e.t == EV_RAISE { b'R' } else { b'M' }, &mut n);
                    let v = e.v as u8;
                    if v >= 10 {
                        put(b'0' + v / 10, &mut n);
                    }
                    put(b'0' + v % 10, &mut n);
                }
                EV_ACCEPT => {
                    put(rel, &mut n);
                    put(b'A', &mut n);
                }
                EV_ROUND => {
                    put(b'|', &mut n);
                    if e.p >= 0 {
                        put(if e.p == me as i8 { b'w' } else { b'l' }, &mut n);
                    }
                }
                _ => {} // deal, run, hand_end: não entram na chave
            }
        }
        put(b':', &mut n);
        put(b"mpr"[self.phase as usize], &mut n);
        n
    }
}

/// Ações abstratas legais (ordenadas) e a ordem de posições usada no mapeamento.
fn abstract_legal(s: &State, abs: &mut [u8; NUM_ACTIONS]) -> (usize, [u8; 3]) {
    let mut conc = [0u8; NUM_ACTIONS];
    let n = s.legal(&mut conc);
    let (order, k) = s.sorted_slots(s.current);
    for i in 0..n {
        let a = conc[i];
        abs[i] = if a >= 6 {
            a
        } else {
            let pos = order[..k].iter().position(|&x| x == a % 3).unwrap() as u8;
            (a / 3) * 3 + pos
        };
    }
    abs[..n].sort_unstable();
    (n, order)
}

#[inline]
fn to_concrete(order: &[u8; 3], ab: u8) -> u8 {
    if ab >= 6 {
        ab
    } else {
        (ab / 3) * 3 + order[(ab % 3) as usize]
    }
}

// ---------------------------------------------------------------------------
// Raiz: mesmo sorteio de `MCCFRTrainer._sample_root`
// ---------------------------------------------------------------------------
fn sample_root(rng: &mut Rng) -> (u8, [u8; 2], [u8; 7]) {
    let dealer = rng.below(2) as u8;
    let r = rng.f64();
    let t = TARGET as u64;
    let (mut a, mut b);
    if r < 0.5 {
        a = rng.below(t);
        b = rng.below(t);
    } else if r < 0.85 {
        a = rng.below(4);
        b = rng.below(4);
    } else {
        a = t - 1;
        b = rng.below(t);
        if rng.f64() < 0.5 {
            std::mem::swap(&mut a, &mut b);
        }
    }
    let mut deck: [u8; 40] = [0; 40];
    for (i, c) in deck.iter_mut().enumerate() {
        *c = i as u8;
    }
    for i in 0..7 {
        let j = i + rng.below((40 - i) as u64) as usize;
        deck.swap(i, j);
    }
    let mut d7 = [0u8; 7];
    d7.copy_from_slice(&deck[..7]);
    (dealer, [a as u8, b as u8], d7)
}

// ---------------------------------------------------------------------------
// Tabela de nós compartilhada
// ---------------------------------------------------------------------------
#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut it = bytes.chunks_exact(8);
        for c in &mut it {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        for &b in it.remainder() {
            self.add(b as u64);
        }
    }
    #[inline]
    fn finish(&self) -> u64 {
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x
    }
}

impl Fx {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x517cc1b727220a95);
    }
}

fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = Fx::default();
    h.write(b);
    h.finish()
}

#[derive(Clone, Default)]
struct Node {
    regret: [f32; NUM_ACTIONS],
    avg: [f64; NUM_ACTIONS], // f64: a média linear soma pesos ~t^2
    visits: u32,
}

type Shard = HashMap<Box<[u8]>, Node, BuildHasherDefault<Fx>>;
const SHARDS: usize = 512;

struct Table {
    shards: Vec<Mutex<Shard>>,
    count: AtomicUsize,
}

impl Table {
    fn new() -> Table {
        Table {
            shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
            count: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn with_node<F: FnOnce(&mut Node)>(&self, key: &[u8], f: F) {
        let h = hash_bytes(key).wrapping_mul(0x9E3779B97F4A7C15);
        let mut sh = self.shards[(h >> 32) as usize % SHARDS].lock().unwrap();
        if let Some(n) = sh.get_mut(key) {
            f(n);
            return;
        }
        let mut n = Node::default();
        f(&mut n);
        sh.insert(key.into(), n);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Remove nós pouco visitados até caber em `target` (como `MCCFRTrainer.prune`).
    fn prune(&self, target: usize) -> u32 {
        let mut k = 2u32;
        loop {
            let mut total = 0;
            for m in &self.shards {
                let mut sh = m.lock().unwrap();
                sh.retain(|_, n| n.visits >= k);
                sh.shrink_to_fit();
                total += sh.len();
            }
            self.count.store(total, Ordering::Relaxed);
            if total <= target {
                return k;
            }
            k *= 2;
        }
    }
}

struct Equity([[f64; 12]; 12]);

impl Equity {
    #[inline]
    fn get(&self, me: u8, opp: u8) -> f64 {
        if me >= TARGET {
            1.0
        } else if opp >= TARGET {
            0.0
        } else {
            self.0[me as usize][opp as usize]
        }
    }
}

struct Trainer {
    table: Table,
    eq: Equity,
    iters: AtomicU64,
    prune_lock: RwLock<()>,
    prune_pending: AtomicBool,
    prunes: Mutex<Vec<(u64, u32, usize)>>,
}

impl Trainer {
    fn traverse(&self, s: &mut State, trav: u8, before: f64, t: f64, rng: &mut Rng) -> f64 {
        if s.terminal() {
            return self.eq.get(s.scores[trav as usize], s.scores[1 - trav as usize]) - before;
        }
        let p = s.current;
        let mut kb = [0u8; 128];
        let kl = s.key(p, &mut kb);
        let key = &kb[..kl];
        let mut abs = [0u8; NUM_ACTIONS];
        let (n, order) = abstract_legal(s, &mut abs);
        let mut sigma = [0f64; NUM_ACTIONS];
        let opp = p != trav;
        self.table.with_node(key, |node| {
            node.visits = node.visits.saturating_add(1);
            let mut tot = 0.0;
            for i in 0..n {
                let r = node.regret[abs[i] as usize].max(0.0) as f64;
                sigma[i] = r;
                tot += r;
            }
            if tot > 0.0 {
                for x in &mut sigma[..n] {
                    *x /= tot;
                }
            } else {
                for x in &mut sigma[..n] {
                    *x = 1.0 / n as f64;
                }
            }
            if opp {
                for i in 0..n {
                    node.avg[abs[i] as usize] += t * sigma[i];
                }
            }
        });
        if opp {
            let i = rng.weighted(&sigma[..n]);
            s.apply(to_concrete(&order, abs[i]));
            return self.traverse(s, trav, before, t, rng);
        }
        let mut vals = [0f64; NUM_ACTIONS];
        let mut v = 0.0;
        for i in 0..n {
            let mut c = *s;
            c.apply(to_concrete(&order, abs[i]));
            vals[i] = self.traverse(&mut c, trav, before, t, rng);
            v += sigma[i] * vals[i];
        }
        self.table.with_node(key, |node| {
            for i in 0..n {
                let a = abs[i] as usize;
                node.regret[a] = (node.regret[a] as f64 + vals[i] - v).max(0.0) as f32; // RM+
            }
        });
        v
    }

    fn worker(&self, seed: u64, deadline: Instant, max_iters: u64, max_nodes: usize) {
        let mut rng = Rng::new(seed);
        loop {
            while self.prune_pending.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(5));
            }
            {
                let _g = self.prune_lock.read().unwrap();
                let it = self.iters.fetch_add(1, Ordering::Relaxed) + 1;
                if it > max_iters {
                    return;
                }
                let (dealer, scores, deck) = sample_root(&mut rng);
                let root = State::dealt(dealer, scores, &deck);
                for trav in 0..2u8 {
                    let before = self.eq.get(scores[trav as usize], scores[1 - trav as usize]);
                    let mut s = root;
                    self.traverse(&mut s, trav, before, it as f64, &mut rng);
                }
                if it % 64 == 0 && Instant::now() >= deadline {
                    return;
                }
            }
            if self.table.count.load(Ordering::Relaxed) > max_nodes
                && !self.prune_pending.swap(true, Ordering::AcqRel)
            {
                let _w = self.prune_lock.write().unwrap();
                if self.table.count.load(Ordering::Relaxed) > max_nodes {
                    let target = (max_nodes as f64 * 0.6) as usize;
                    let k = self.table.prune(target);
                    let it = self.iters.load(Ordering::Relaxed);
                    let left = self.table.count.load(Ordering::Relaxed);
                    eprintln!("poda: it={} k={} nós={}", it, k, left);
                    self.prunes.lock().unwrap().push((it, k, left));
                }
                self.prune_pending.store(false, Ordering::Release);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: Option<T>) -> T {
    match args.iter().position(|a| a == name) {
        Some(i) => args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("valor inválido para {}", name)),
        None => default.unwrap_or_else(|| panic!("falta {}", name)),
    }
}

fn load_equity(path: &str) -> Equity {
    let text = std::fs::read_to_string(path).expect("tabela de equidade");
    let vals: Vec<f64> = text.split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(vals.len(), 144, "tabela de equidade deve ter 12x12 valores");
    let mut t = [[0f64; 12]; 12];
    for a in 0..12 {
        for b in 0..12 {
            t[a][b] = vals[a * 12 + b];
        }
    }
    Equity(t)
}

fn train(args: &[String]) {
    let eq = load_equity(&arg::<String>(args, "--eq", None));
    let out: String = arg(args, "--out", None);
    let summary: String = arg(args, "--summary", None);
    let seed: u64 = arg(args, "--seed", Some(0));
    let threads: u64 = arg(args, "--threads", Some(4));
    let seconds: f64 = arg(args, "--seconds", Some(60.0));
    let max_iters: u64 = arg(args, "--iterations", Some(u64::MAX / 2));
    let max_nodes: usize = arg(args, "--max-nodes", Some(6_000_000));
    let save_min: u32 = arg(args, "--save-min-visits", Some(100));
    let log_every: f64 = arg(args, "--log-every", Some(60.0));

    let tr = Trainer {
        table: Table::new(),
        eq,
        iters: AtomicU64::new(0),
        prune_lock: RwLock::new(()),
        prune_pending: AtomicBool::new(false),
        prunes: Mutex::new(Vec::new()),
    };
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_secs_f64(seconds);
    let done = AtomicBool::new(false);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|i| {
                let tr = &tr;
                sc.spawn(move || tr.worker(seed.wrapping_mul(1_000_003).wrapping_add(i), deadline, max_iters, max_nodes))
            })
            .collect();
        let tr = &tr;
        let done = &done;
        sc.spawn(move || {
            let mut next = log_every;
            while !done.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                let el = t0.elapsed().as_secs_f64();
                if el >= next {
                    next += log_every;
                    let it = tr.iters.load(Ordering::Relaxed);
                    println!(
                        "t={:.0}s it={} nós={} {:.0} it/s",
                        el,
                        it,
                        tr.table.count.load(Ordering::Relaxed),
                        it as f64 / el
                    );
                    std::io::stdout().flush().ok();
                }
            }
        });
        for h in hs {
            h.join().unwrap();
        }
        done.store(true, Ordering::Relaxed);
    });
    let elapsed = t0.elapsed().as_secs_f64();
    let iterations = tr.iters.load(Ordering::Relaxed).min(max_iters);

    // Histograma de visitas e gravação da política média.
    let thresholds = [1u32, 10, 30, 100, 300, 1000, 2000, 5000, 10000, 100000];
    let mut hist = [0usize; 10];
    let mut saved = 0usize;
    let f = std::fs::File::create(&out).expect("arquivo de saída");
    let mut w = BufWriter::new(f);
    for m in &tr.table.shards {
        let sh = m.lock().unwrap();
        for (k, n) in sh.iter() {
            for (i, &th) in thresholds.iter().enumerate() {
                if n.visits >= th {
                    hist[i] += 1;
                }
            }
            if n.visits < save_min {
                continue;
            }
            let tot: f64 = n.avg.iter().sum();
            if tot <= 0.0 {
                continue;
            }
            w.write_all(k).unwrap();
            write!(w, "\t{}\t", n.visits).unwrap();
            let mut first = true;
            for (a, &x) in n.avg.iter().enumerate() {
                if x > 0.0 {
                    if !first {
                        w.write_all(b",").unwrap();
                    }
                    write!(w, "{}={}", a, x / tot).unwrap();
                    first = false;
                }
            }
            w.write_all(b"\n").unwrap();
            saved += 1;
        }
    }
    w.flush().unwrap();
    let hist_json: Vec<String> = thresholds
        .iter()
        .zip(hist.iter())
        .map(|(t, c)| format!("\"{}\": {}", t, c))
        .collect();
    let prunes: Vec<String> = tr
        .prunes
        .lock()
        .unwrap()
        .iter()
        .map(|(i, k, l)| format!("[{}, {}, {}]", i, k, l))
        .collect();
    let js = format!(
        "{{\n  \"rules_version\": \"{}\",\n  \"iterations\": {},\n  \"elapsed_s\": {:.1},\n  \"it_per_s\": {:.1},\n  \"threads\": {},\n  \"seed\": {},\n  \"nodes_final\": {},\n  \"saved_nodes\": {},\n  \"save_min_visits\": {},\n  \"max_nodes\": {},\n  \"visits_at_least\": {{{}}},\n  \"prunes_it_k_left\": [{}]\n}}\n",
        rules_version(),
        iterations,
        elapsed,
        iterations as f64 / elapsed,
        threads,
        seed,
        tr.table.count.load(Ordering::Relaxed),
        saved,
        save_min,
        max_nodes,
        hist_json.join(", "),
        prunes.join(", ")
    );
    std::fs::write(&summary, js).unwrap();
    println!("ok: it={} {:.0} it/s nós salvos={}", iterations, iterations as f64 / elapsed, saved);
}

/// Jogos com ações aleatórias, impressos para comparação com o motor Python.
///   G dealer s0 s1 c0..c6
///   D jogador chave abstratas(csv) ação_concreta
///   E s0 s1 fase
fn selftest(args: &[String]) {
    let seed: u64 = arg(args, "--seed", Some(0));
    let games: u64 = arg(args, "--games", Some(1000));
    let mut rng = Rng::new(seed);
    let stdout = std::io::stdout();
    let mut w = BufWriter::new(stdout.lock());
    for _ in 0..games {
        let (dealer, scores, deck) = sample_root(&mut rng);
        let mut s = State::dealt(dealer, scores, &deck);
        write!(w, "G {} {} {}", dealer, scores[0], scores[1]).unwrap();
        for c in deck {
            write!(w, " {}", c).unwrap();
        }
        writeln!(w).unwrap();
        while !s.terminal() {
            let mut kb = [0u8; 128];
            let kl = s.key(s.current, &mut kb);
            let mut abs = [0u8; NUM_ACTIONS];
            let (n, order) = abstract_legal(&s, &mut abs);
            let ab = abs[rng.below(n as u64) as usize];
            let a = to_concrete(&order, ab);
            let csv: Vec<String> = abs[..n].iter().map(|x| x.to_string()).collect();
            writeln!(w, "D {} {} {} {} {}", s.current, std::str::from_utf8(&kb[..kl]).unwrap(), csv.join(","), a,
                     v2::key_hash_v2(&kb[..kl])).unwrap();
            s.apply(a);
        }
        writeln!(w, "E {} {} {}", s.scores[0], s.scores[1], if s.phase == PH_MATCH_OVER { "match_over" } else { "hand_over" }).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--rules").and_then(|i| args.get(i + 1)).map(|s| s.as_str()) {
        None | Some("h0") => {}
        Some("o1") => O1.store(true, Ordering::Relaxed),
        Some(x) => {
            eprintln!("regulamento desconhecido: {} (use h0 ou o1)", x);
            std::process::exit(2);
        }
    }
    match args.iter().position(|a| a == "--abstraction").and_then(|i| args.get(i + 1)).map(|s| s.as_str()) {
        None | Some("v1") => {}
        Some("v2a") => ABSV.store(1, Ordering::Relaxed),
        Some("v2b") => ABSV.store(2, Ordering::Relaxed),
        Some(x) => {
            eprintln!("abstração desconhecida: {} (use v1, v2a ou v2b)", x);
            std::process::exit(2);
        }
    }
    match args.get(1).map(|s| s.as_str()) {
        Some("train") => train(&args),
        Some("train2") => v2::train2(&args),
        Some("selftest") => selftest(&args),
        Some("version") => println!("{}", rules_version()),
        _ => {
            eprintln!("uso: mccfr train|selftest|version ...");
            std::process::exit(2);
        }
    }
}
