//! Núcleo v2 do ES-MCCFR: mesmo algoritmo e mesma regra de atualização do v1
//! (amostragem externa, RM+, média linear, utilidade = Δ equidade), com outra
//! estrutura de dados:
//!
//! * **Tabela sem travas** (open addressing, pré-alocada, zerada pelo SO): cada
//!   entrada guarda só o hash de 64 bits da chave do infoset — nada de `Box<[u8]>`
//!   nem `Mutex` por consulta.
//! * **Uma consulta por nó**: o índice obtido ao ler a estratégia é reutilizado
//!   para gravar o arrependimento (o v1 fazia duas consultas com trava).
//! * **Atualizações "Hogwild"**: leituras/escritas atômicas relaxadas, sem CAS
//!   nos valores (técnica padrão de MCCFR/SGD paralelo; perdas raras de update).
//! * **Sem poda que para o mundo**: quando a tabela atinge o limite, nós novos
//!   deixam de ser inseridos (jogam com estratégia uniforme, sem atualização),
//!   em vez de varrer e apagar milhões de nós com todas as threads paradas.
//! * Exporta direto no formato compacto (`.npy`), sem texto.
//!
//! Hash da chave (`key_hash_v2`): FNV-1a 64 sobre os bytes da chave de
//! `info_key` + finalizador splitmix64; nunca 0 (0 marca entrada vazia).
//! Implementado igual em Python (`trucoai.cfr.compact.key_hash_fnv`) e no kernel GPU.

use super::*;
use std::alloc::{alloc_zeroed, Layout};
use std::sync::atomic::AtomicU8;

const MAX_PROBE: usize = 64;

#[inline]
pub fn key_hash_v2(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    if h == 0 { 1 } else { h }
}

#[repr(C, align(64))]
pub struct Slot {
    pub key: AtomicU64,
    pub visits: AtomicU32,
    pub regret: [AtomicU32; NUM_ACTIONS], // f32 em bits
    pub avg: [AtomicU64; NUM_ACTIONS],    // f64 em bits (média linear soma ~t²)
}

pub struct Table2 {
    slots: *mut Slot,
    // Filtro de admissão: contador aproximado por hash (u8 saturado). Um nó só
    // entra na tabela depois de ``gate`` visitas; antes disso joga uniforme e não
    // acumula — substitui a poda periódica do v1 sem parar as threads.
    sketch: Vec<AtomicU8>,
    sketch_mask: usize,
    gate: u8,
    mask: usize,
    pub cap: usize,
    pub limit: usize,
    pub count: AtomicUsize,
    pub dropped: AtomicU64, // inserções recusadas por tabela cheia
}

unsafe impl Sync for Table2 {}
unsafe impl Send for Table2 {}

impl Table2 {
    pub fn new(log2_cap: u32, load: f64, log2_sketch: u32, gate: u8) -> Table2 {
        let cap = 1usize << log2_cap;
        let layout = Layout::array::<Slot>(cap).expect("layout");
        // Memória zerada pelo SO sob demanda: todos os campos começam em 0.
        let ptr = unsafe { alloc_zeroed(layout) } as *mut Slot;
        assert!(!ptr.is_null(), "sem memória para {} entradas", cap);
        let sk = 1usize << log2_sketch;
        Table2 {
            slots: ptr,
            sketch: (0..sk).map(|_| AtomicU8::new(0)).collect(),
            sketch_mask: sk - 1,
            gate,
            mask: cap - 1,
            cap,
            limit: (cap as f64 * load) as usize,
            count: AtomicUsize::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn slot(&self, i: usize) -> &Slot {
        unsafe { &*self.slots.add(i) }
    }

    /// Índice da entrada de `h`, inserindo se preciso. `None` = tabela cheia.
    #[inline]
    pub fn find(&self, h: u64) -> Option<usize> {
        let mut i = (h as usize) & self.mask;
        for _ in 0..MAX_PROBE {
            let s = self.slot(i);
            let k = s.key.load(Ordering::Relaxed);
            if k == h {
                return Some(i);
            }
            if k == 0 {
                if self.gate > 1 {
                    let c = &self.sketch[((h >> 17) as usize) & self.sketch_mask];
                    let v = c.load(Ordering::Relaxed);
                    if v < self.gate - 1 {
                        c.store(v + 1, Ordering::Relaxed);
                        return None; // ainda não admitido
                    }
                }
                if self.count.load(Ordering::Relaxed) >= self.limit {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                match s.key.compare_exchange(0, h, Ordering::AcqRel, Ordering::Relaxed) {
                    Ok(_) => {
                        self.count.fetch_add(1, Ordering::Relaxed);
                        return Some(i);
                    }
                    Err(actual) if actual == h => return Some(i),
                    Err(_) => {}
                }
            }
            i = (i + 1) & self.mask;
        }
        self.dropped.fetch_add(1, Ordering::Relaxed);
        None
    }
}

#[inline]
fn ld32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}
#[inline]
fn st32(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed)
}
#[inline]
fn ld64(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::Relaxed))
}
#[inline]
fn st64(a: &AtomicU64, v: f64) {
    a.store(v.to_bits(), Ordering::Relaxed)
}

thread_local! {
    // Contador de nós por thread (um atômico global disputado custava ~25% de vazão).
    static NODES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub struct Trainer2 {
    pub table: Table2,
    pub eq: Equity,
    pub iters: AtomicU64,
    // Poda por arrependimento (estilo Pluribus): depois de ``prune_after``
    // iterações, ações com arrependimento RM+ = 0 (prob. 0 na estratégia atual) são
    // exploradas só com prob. ``1 - prune_p`` nos nós de quem treina. Como σ=0, o
    // valor do nó não muda; só atrasa a "recuperação" de ações ruins.
    pub prune_p: f64,
    pub prune_after: u64,
    pub nodes: AtomicU64, // nós visitados (medição)
}

impl Trainer2 {
    fn traverse(&self, s: &mut State, trav: u8, before: f64, t: f64, rng: &mut Rng) -> f64 {
        if s.terminal() {
            return self.eq.get(s.scores[trav as usize], s.scores[1 - trav as usize]) - before;
        }
        let p = s.current;
        let mut kb = [0u8; 128];
        let kl = s.key(p, &mut kb);
        let idx = self.table.find(key_hash_v2(&kb[..kl]));
        NODES.with(|c| c.set(c.get() + 1));
        let mut abs = [0u8; NUM_ACTIONS];
        let (n, order) = abstract_legal(s, &mut abs);
        let mut sigma = [0f64; NUM_ACTIONS];
        let opp = p != trav;
        let mut tot = 0.0;
        if let Some(i) = idx {
            let sl = self.table.slot(i);
            sl.visits.store(sl.visits.load(Ordering::Relaxed).saturating_add(1), Ordering::Relaxed);
            for k in 0..n {
                let r = ld32(&sl.regret[abs[k] as usize]).max(0.0) as f64;
                sigma[k] = r;
                tot += r;
            }
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
            if let Some(i) = idx {
                let sl = self.table.slot(i);
                for k in 0..n {
                    let a = &sl.avg[abs[k] as usize];
                    st64(a, ld64(a) + t * sigma[k]);
                }
            }
            let k = rng.weighted(&sigma[..n]);
            s.apply(to_concrete(&order, abs[k]));
            return self.traverse(s, trav, before, t, rng);
        }
        let mut vals = [0f64; NUM_ACTIONS];
        let mut v = 0.0;
        let mut skip = [false; NUM_ACTIONS];
        if self.prune_p > 0.0 && t as u64 > self.prune_after && tot > 0.0 {
            for k in 0..n {
                skip[k] = sigma[k] == 0.0 && rng.f64() < self.prune_p;
            }
        }
        for k in 0..n {
            if skip[k] {
                continue; // σ_k = 0: não contribui para v; arrependimento fica como está
            }
            let mut c = *s;
            c.apply(to_concrete(&order, abs[k]));
            vals[k] = self.traverse(&mut c, trav, before, t, rng);
            v += sigma[k] * vals[k];
        }
        if let Some(i) = idx {
            let sl = self.table.slot(i);
            for k in 0..n {
                if skip[k] {
                    continue;
                }
                let r = &sl.regret[abs[k] as usize];
                st32(r, (ld32(r) as f64 + vals[k] - v).max(0.0) as f32); // RM+
            }
        }
        v
    }

    fn worker(&self, seed: u64, deadline: Instant, max_iters: u64) {
        let mut rng = Rng::new(seed);
        loop {
            let it = self.iters.fetch_add(1, Ordering::Relaxed) + 1;
            if it > max_iters {
                self.nodes.fetch_add(NODES.with(|c| c.replace(0)), Ordering::Relaxed);
                return;
            }
            let (dealer, scores, deck) = sample_root(&mut rng);
            let root = State::dealt(dealer, scores, &deck);
            for trav in 0..2u8 {
                let before = self.eq.get(scores[trav as usize], scores[1 - trav as usize]);
                let mut s = root;
                self.traverse(&mut s, trav, before, it as f64, &mut rng);
            }
            if it % 256 == 0 {
                self.nodes.fetch_add(NODES.with(|c| c.replace(0)), Ordering::Relaxed);
                if Instant::now() >= deadline {
                    return;
                }
            }
        }
    }
}

fn write_npy(path: &str, dtype: &str, shape: &str, bytes: &[u8]) {
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", dtype, shape);
    let total = 10 + header.len() + 1;
    let pad = (64 - total % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut f = BufWriter::new(std::fs::File::create(path).expect("arquivo .npy"));
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(bytes).unwrap();
    f.flush().unwrap();
}

/// `mccfr train2 --eq T --out-dir D --summary S --seed N --threads N --seconds X
///   [--iterations N] [--log2-cap 25] [--load 0.75] [--save-min-visits 30]`
/// (filtro de admissão: ``--gate 4 --log2-sketch 28``; ``--gate 1`` desliga)
pub fn train2(args: &[String]) {
    let eq = load_equity(&arg::<String>(args, "--eq", None));
    let out_dir: String = arg(args, "--out-dir", None);
    let summary: String = arg(args, "--summary", None);
    let seed: u64 = arg(args, "--seed", Some(0));
    let threads: u64 = arg(args, "--threads", Some(16));
    let seconds: f64 = arg(args, "--seconds", Some(60.0));
    let max_iters: u64 = arg(args, "--iterations", Some(u64::MAX / 2));
    let log2_cap: u32 = arg(args, "--log2-cap", Some(25));
    let load: f64 = arg(args, "--load", Some(0.75));
    let save_min: u32 = arg(args, "--save-min-visits", Some(30));
    let log2_sketch: u32 = arg(args, "--log2-sketch", Some(28));
    let gate: u8 = arg(args, "--gate", Some(4));
    let prune_p: f64 = arg(args, "--prune", Some(0.0));
    let prune_after: u64 = arg(args, "--prune-after", Some(1_000_000));
    let log_every: f64 = arg(args, "--log-every", Some(60.0));

    let tr = Trainer2 { table: Table2::new(log2_cap, load, log2_sketch, gate), eq, iters: AtomicU64::new(0),
                       prune_p, prune_after, nodes: AtomicU64::new(0) };
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_secs_f64(seconds);
    let done = AtomicBool::new(false);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|i| {
                let tr = &tr;
                sc.spawn(move || tr.worker(seed.wrapping_mul(1_000_003).wrapping_add(i), deadline, max_iters))
            })
            .collect();
        let (tr, done) = (&tr, &done);
        sc.spawn(move || {
            let mut next = log_every;
            while !done.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                let el = t0.elapsed().as_secs_f64();
                if el >= next {
                    next += log_every;
                    let it = tr.iters.load(Ordering::Relaxed);
                    println!("t={:.0}s it={} nós={} recusados={} {:.0} it/s {:.0} nós/it", el, it,
                             tr.table.count.load(Ordering::Relaxed),
                             tr.table.dropped.load(Ordering::Relaxed), it as f64 / el,
                             tr.nodes.load(Ordering::Relaxed) as f64 / it.max(1) as f64);
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

    // Exportação compacta: keys u64, visits u32, probs u8 [N, 9] (não ordenado).
    let (mut keys, mut visits, mut probs) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..tr.table.cap {
        let sl = tr.table.slot(i);
        let k = sl.key.load(Ordering::Relaxed);
        let v = sl.visits.load(Ordering::Relaxed);
        if k == 0 || v < save_min {
            continue;
        }
        let avg: Vec<f64> = sl.avg.iter().map(ld64).collect();
        let tot: f64 = avg.iter().sum();
        if tot <= 0.0 {
            continue;
        }
        let mut q: Vec<u8> = avg.iter().map(|x| ((x / tot) * 255.0).round() as u8).collect();
        if q.iter().all(|&x| x == 0) {
            let best = (0..NUM_ACTIONS).max_by(|&a, &b| avg[a].partial_cmp(&avg[b]).unwrap()).unwrap();
            q[best] = 1;
        }
        keys.extend_from_slice(&k.to_le_bytes());
        visits.extend_from_slice(&v.to_le_bytes());
        probs.extend_from_slice(&q);
    }
    let n = visits.len() / 4;
    std::fs::create_dir_all(&out_dir).unwrap();
    write_npy(&format!("{}/keys_unsorted.npy", out_dir), "<u8", &format!("({},)", n), &keys);
    write_npy(&format!("{}/visits_unsorted.npy", out_dir), "<u4", &format!("({},)", n), &visits);
    write_npy(&format!("{}/probs_unsorted.npy", out_dir), "|u1", &format!("({}, {})", n, NUM_ACTIONS), &probs);
    let js = format!(
        "{{\n  \"kernel\": \"cpu-v2\",\n  \"abstraction\": \"{}\",\n  \"rules_version\": \"{}\",\n  \"iterations\": {},\n  \"elapsed_s\": {:.1},\n  \"it_per_s\": {:.1},\n  \"threads\": {},\n  \"seed\": {},\n  \"capacity\": {},\n  \"nodes_final\": {},\n  \"dropped_inserts\": {},\n  \"saved_nodes\": {},\n  \"save_min_visits\": {},\n  \"key_hash\": \"fnv1a64-splitmix\",\n  \"gate\": {},\n  \"prune\": {},\n  \"prune_after\": {},\n  \"nodes_per_iter\": {:.1}\n}}\n",
        abs_name(), rules_version(), iterations, elapsed, iterations as f64 / elapsed, threads, seed,
        tr.table.cap, tr.table.count.load(Ordering::Relaxed),
        tr.table.dropped.load(Ordering::Relaxed), n, save_min, gate, prune_p, prune_after,
        tr.nodes.load(Ordering::Relaxed) as f64 / iterations.max(1) as f64
    );
    std::fs::write(&summary, js).unwrap();
    println!("ok: it={} {:.0} it/s nós={} salvos={}", iterations, iterations as f64 / elapsed,
             tr.table.count.load(Ordering::Relaxed), n);
}
