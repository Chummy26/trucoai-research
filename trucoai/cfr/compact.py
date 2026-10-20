"""Blueprint compacto: política média do MCCFR em arrays numpy mapeados em memória.

O formato pickle (dict de strings) ocupa ~480 bytes por nó em RAM; 1,35 M de
nós já davam ~650 MB por processo. Aqui cada nó ocupa 21 bytes em disco e a
leitura é por ``np.load(mmap_mode="r")``: processos diferentes compartilham as
mesmas páginas do sistema operacional.

Diretório ``<run>/compact/``:
    keys.npy    uint64 ordenado — blake2b-64 da chave ``info_key``
    probs.npy   uint8 [N, 9]    — probabilidade de cada ação abstrata × 255
    visits.npy  uint32 [N]      — visitas do nó no treino
    meta.json   metadados do treino (regras, iterações, sementes...)

Colisão de hash: com N = 10^7, P(alguma colisão) ≈ N²/2^65 ≈ 3·10^-6.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

NUM_ACTIONS = 9


def key_hash(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


_M64 = (1 << 64) - 1


def key_hash_fnv(key: str) -> int:
    """FNV-1a 64 + finalizador splitmix64 (nunca 0). Igual a ``key_hash_v2`` do
    núcleo Rust v2 e do kernel GPU, que guardam só este hash na tabela."""
    h = 0xcbf29ce484222325
    for b in key.encode():
        h = ((h ^ b) * 0x100000001b3) & _M64
    h = ((h ^ (h >> 30)) * 0xbf58476d1ce4e5b9) & _M64
    h = ((h ^ (h >> 27)) * 0x94d049bb133111eb) & _M64
    h ^= h >> 31
    return h or 1


HASHES = {"blake2b64": key_hash, "fnv1a64-splitmix": key_hash_fnv}


class CompactPolicy:
    def __init__(self, directory: Path, cache_size: int = 200_000):
        d = Path(directory)
        self.keys = np.load(d / "keys.npy", mmap_mode="r")
        self.probs = np.load(d / "probs.npy", mmap_mode="r")
        self.visits_arr = np.load(d / "visits.npy", mmap_mode="r")
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self._hash = HASHES[self.meta.get("key_hash", "blake2b64")]
        self._cache: "OrderedDict[str, Tuple[Optional[Dict[int, float]], int]]" = OrderedDict()
        self._cache_size = cache_size

    def __len__(self) -> int:
        return len(self.keys)

    def lookup(self, key: str) -> Tuple[Optional[Dict[int, float]], int]:
        """``(distribuição sobre ações abstratas, visitas)``; ``(None, 0)`` se ausente."""
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        h = np.uint64(self._hash(key))
        i = int(np.searchsorted(self.keys, h))
        if i < len(self.keys) and self.keys[i] == h:
            row = self.probs[i]
            tot = int(row.sum())
            dist = {a: int(p) / tot for a, p in enumerate(row) if p} if tot else None
            res = (dist, int(self.visits_arr[i]))
        else:
            res = (None, 0)
        self._cache[key] = res
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return res


def _parse_tsv_line(line: str):
    key, v, dist = line.rstrip("\n").split("\t")
    return key, int(v), dist


def build_from_rows(rows: Iterable[Tuple[str, int, Dict[int, float]]], n: int, out: Path,
                    meta: dict) -> int:
    """Grava o formato compacto a partir de ``n`` linhas ``(chave, visitas, dist)``."""
    keys = np.empty(n, dtype=np.uint64)
    probs = np.zeros((n, NUM_ACTIONS), dtype=np.uint8)
    visits = np.empty(n, dtype=np.uint32)
    i = 0
    for key, v, dist in rows:
        keys[i] = key_hash(key)
        visits[i] = min(v, 2**32 - 1)
        q = np.zeros(NUM_ACTIONS)
        for a, p in dist.items():
            q[int(a)] = p
        q = np.round(q / max(q.sum(), 1e-12) * 255)
        if q.sum() == 0:  # todas as probabilidades pequenas: mantém a maior
            q[int(max(dist, key=dist.get))] = 1
        probs[i] = q.astype(np.uint8)
        i += 1
    keys, probs, visits = keys[:i], probs[:i], visits[:i]
    order = np.argsort(keys, kind="stable")
    keys, probs, visits = keys[order], probs[order], visits[order]
    dup = np.count_nonzero(keys[1:] == keys[:-1])
    if dup:
        raise ValueError(f"{dup} colisões de hash")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "keys.npy", keys)
    np.save(out / "probs.npy", probs)
    np.save(out / "visits.npy", visits)
    (out / "meta.json").write_text(json.dumps({**meta, "nodes": int(i)}, indent=2,
                                              ensure_ascii=False), encoding="utf-8")
    return i


def build_from_tsv(tsv: Path, out: Path, min_visits: int, meta: dict) -> int:
    """Converte ``policy.tsv`` do núcleo Rust em duas passadas (memória limitada)."""
    n = 0
    with open(tsv, encoding="ascii") as f:
        for line in f:
            if int(line.split("\t", 2)[1]) >= min_visits:
                n += 1

    def rows():
        with open(tsv, encoding="ascii") as f:
            for line in f:
                key, v, dist = _parse_tsv_line(line)
                if v >= min_visits:
                    yield key, v, {int(a): float(p) for a, p in (x.split("=") for x in dist.split(","))}

    return build_from_rows(rows(), n, out, {**meta, "min_visits": min_visits})


def build_from_pkl(pkl: Path, out: Path, min_visits: int = 1) -> int:
    import pickle
    with open(pkl, "rb") as f:
        blob = pickle.load(f)
    items = [(k, int(blob["visits"].get(k, 0)), d) for k, d in blob["policy"].items()
             if blob["visits"].get(k, 0) >= min_visits]
    return build_from_rows(items, len(items), out, {**blob["meta"], "min_visits": min_visits,
                                                    "source": str(pkl)})


def finalize_unsorted(directory: Path, meta: dict) -> int:
    """Ordena a exportação do núcleo v2/GPU (``*_unsorted.npy``) no formato compacto."""
    d = Path(directory)
    keys = np.load(d / "keys_unsorted.npy")
    visits = np.load(d / "visits_unsorted.npy")
    probs = np.load(d / "probs_unsorted.npy")
    order = np.argsort(keys, kind="stable")
    keys, visits, probs = keys[order], visits[order], probs[order]
    if np.count_nonzero(keys[1:] == keys[:-1]):
        raise ValueError("chaves repetidas na exportação")
    np.save(d / "keys.npy", keys)
    np.save(d / "probs.npy", probs)
    np.save(d / "visits.npy", visits)
    (d / "meta.json").write_text(json.dumps({**meta, "nodes": int(len(keys)),
                                             "key_hash": "fnv1a64-splitmix"}, indent=2,
                                            ensure_ascii=False), encoding="utf-8")
    for f in ("keys_unsorted.npy", "visits_unsorted.npy", "probs_unsorted.npy"):
        (d / f).unlink()
    return int(len(keys))
