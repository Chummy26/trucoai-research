"""Treinador ES-MCCFR na GPU (kernel ``mccfr_train``) e exportação compacta.

Mesma regra de atualização do núcleo Rust (amostragem externa, RM+, média
linear, utilidade = Δ equidade), mesma chave (hash FNV) e mesma tabela v2.
Lançamentos curtos (< ~0,5 s) por causa do watchdog do Windows em placas de vídeo.
"""

from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .cuda import Kernel, ptr

SLOT_BYTES = 128


class GpuTrainer:
    def __init__(self, equity_txt: Path, o1: bool = True, log2_cap: int = 25, load: float = 0.75,
                 log2_sketch: int = 28, gate: int = 4, seed: int = 0, block: int = 128,
                 blocks: Optional[int] = None, stack: int = 8192, abstraction: str = "v1"):
        self.k = Kernel()
        self.k.set_stack(stack)
        self.cap = 1 << log2_cap
        self.table = torch.zeros(self.cap * SLOT_BYTES // 8, dtype=torch.int64, device="cuda")
        self.sketch = torch.zeros(1 << log2_sketch, dtype=torch.uint8, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.dropped = torch.zeros(1, dtype=torch.int64, device="cuda")
        vals = [float(x) for x in Path(equity_txt).read_text().split()]
        assert len(vals) == 144
        self.eq = torch.tensor(vals, dtype=torch.float64, device="cuda")
        self.limit = int(self.cap * load)
        self.log2_sketch, self.gate, self.o1, self.seed = log2_sketch, gate, int(o1), seed
        self.abstraction = abstraction
        self.absv = {"v1": 0, "v2a": 1, "v2b": 2}[abstraction]
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.block = block
        self.blocks = blocks or sms * 8
        self.iters = 0

    @property
    def nthreads(self) -> int:
        return self.blocks * self.block

    def step(self, iters_per_thread: int = 1) -> None:
        self.k.launch("mccfr_train", self.blocks, self.block, [
            ptr(self.table), ctypes.c_uint64(self.cap - 1), ctypes.c_uint64(self.limit),
            ptr(self.sketch), ctypes.c_uint64((1 << self.log2_sketch) - 1), ctypes.c_int(self.gate),
            ptr(self.count), ptr(self.dropped), ptr(self.eq), ctypes.c_int(self.o1),
            ctypes.c_uint64(self.seed), ctypes.c_uint64(self.iters), ctypes.c_int(iters_per_thread),
            ctypes.c_int(self.absv)])
        self.iters += iters_per_thread * self.nthreads

    def train(self, seconds: float, log_every: float = 60.0, iters_per_thread: int = 1,
              max_iters: Optional[int] = None) -> dict:
        t0 = time.time()
        nxt = log_every
        while time.time() - t0 < seconds and (max_iters is None or self.iters < max_iters):
            self.step(iters_per_thread)
            self.k.sync()
            el = time.time() - t0
            if el >= nxt:
                nxt += log_every
                print(f"[gpu] t={el:.0f}s it={self.iters} nós={int(self.count.item())} "
                      f"recusados={int(self.dropped.item())} {self.iters / el:.0f} it/s", flush=True)
        el = time.time() - t0
        return {"iterations": self.iters, "elapsed_s": round(el, 1), "it_per_s": round(self.iters / el, 1)}

    def export_unsorted(self, out_dir: Path, save_min: int = 30) -> int:
        """Grava ``keys/visits/probs_unsorted.npy`` como o núcleo Rust v2."""
        t64 = self.table.view(self.cap, 16)
        keys = t64[:, 0]
        visits = (t64[:, 1] & 0xFFFFFFFF)
        avg = self.table.view(torch.float64).view(self.cap, 16)[:, 6:15]
        tot = avg.sum(1)
        m = (keys != 0) & (visits >= save_min) & (tot > 0)
        k, v, a, s = keys[m], visits[m], avg[m], tot[m]
        q = torch.round(a / s[:, None] * 255).clamp(0, 255).to(torch.uint8)
        zero = q.sum(1) == 0
        if zero.any():
            q[zero, a[zero].argmax(1)] = 1
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "keys_unsorted.npy", k.cpu().numpy().view(np.uint64))
        np.save(out / "visits_unsorted.npy", v.to(torch.int64).cpu().numpy().astype(np.uint32))
        np.save(out / "probs_unsorted.npy", q.cpu().numpy())
        return int(m.sum().item())

    def summary(self) -> dict:
        return {"kernel": "gpu-cuda", "device": torch.cuda.get_device_name(0),
                "capacity": self.cap, "nodes_final": int(self.count.item()),
                "dropped_inserts": int(self.dropped.item()), "gate": self.gate,
                "threads": self.nthreads, "key_hash": "fnv1a64-splitmix",
                "rules_version": "paulista-1v1-o1" if self.o1 else "paulista-1v1-h0",
                "abstraction": self.abstraction}


ST_BYTES, META_BYTES = 144, 64


class WavefrontTrainer(GpuTrainer):
    """ES-MCCFR em ondas: uma thread por nó de cada profundidade, para ``batch``
    mãos (2·batch árvores) ao mesmo tempo. Mesma matemática do kernel recursivo."""

    def __init__(self, *a, batch: int = 16384, prune: float = 0.0, prune_after: int = 2_000_000, **kw):
        super().__init__(*a, **kw)
        self.batch = batch
        self.prune, self.prune_after = prune, prune_after
        self.nodes = 0
        self._batches = 0
        self.max_level_nodes = 0

    def _grid(self, n: int):
        return (n + self.block - 1) // self.block

    def step(self, iters_per_thread: int = 1) -> None:  # um lote de ``batch`` mãos
        B = self.batch
        dev = "cuda"
        N = 2 * B
        states = torch.empty((N, ST_BYTES), dtype=torch.uint8, device=dev)
        root_of = torch.empty(N, dtype=torch.int32, device=dev)
        rtrav = torch.empty(N, dtype=torch.uint8, device=dev)
        rbefore = torch.empty(N, dtype=torch.float64, device=dev)
        rt = torch.empty(N, dtype=torch.float64, device=dev)
        self.k.launch("wf_roots", self._grid(B), self.block, [
            ctypes.c_int(B), ctypes.c_uint64(self.seed), ctypes.c_uint64(self.iters),
            ptr(states), ptr(root_of), ptr(rtrav), ptr(rbefore), ptr(rt), ptr(self.eq)])
        levels = []
        lvl = 0
        while N > 0:
            meta = torch.empty((N, META_BYTES), dtype=torch.uint8, device=dev)
            nchild = torch.empty(N, dtype=torch.int32, device=dev)
            val = torch.zeros(N, dtype=torch.float64, device=dev)
            rng_key = (self.seed * 0x9E3779B97F4A7C15 + self._batches * 0xD1B54A32D192ED03
                       + lvl * 0x94D049BB133111EB) & ((1 << 64) - 1)
            self.k.launch("wf_expand", self._grid(N), self.block, [
                ctypes.c_int(N), ptr(states), ptr(root_of), ptr(rtrav), ptr(rbefore), ptr(rt),
                ptr(meta), ptr(nchild), ptr(val), ptr(self.table), ctypes.c_uint64(self.cap - 1),
                ctypes.c_uint64(self.limit), ptr(self.sketch),
                ctypes.c_uint64((1 << self.log2_sketch) - 1), ctypes.c_int(self.gate),
                ptr(self.count), ptr(self.dropped), ptr(self.eq), ctypes.c_int(self.o1),
                ctypes.c_uint64(rng_key), ctypes.c_double(self.prune),
                ctypes.c_double(float(self.prune_after)), ctypes.c_int(self.absv)])
            self.nodes += N
            incl = torch.cumsum(nchild, 0, dtype=torch.int32)
            total = int(incl[-1].item())
            offs = incl - nchild
            levels.append((N, meta, nchild, offs, val))
            self.max_level_nodes = max(self.max_level_nodes, N)
            if total == 0:
                break
            nstates = torch.empty((total, ST_BYTES), dtype=torch.uint8, device=dev)
            nroot = torch.empty(total, dtype=torch.int32, device=dev)
            parent = torch.repeat_interleave(torch.arange(N, device=dev), nchild, output_size=total)
            self.k.launch("wf_emit2", self._grid(total), self.block, [
                ctypes.c_int(total), ptr(states), ptr(root_of), ptr(meta), ptr(offs), ptr(parent),
                ptr(nstates), ptr(nroot)])
            states, root_of, N = nstates, nroot, total
            lvl += 1
        for L in range(len(levels) - 2, -1, -1):
            N, meta, nchild, offs, val = levels[L]
            child_val = levels[L + 1][4]
            self.k.launch("wf_backup", self._grid(N), self.block, [
                ctypes.c_int(N), ptr(meta), ptr(nchild), ptr(offs), ptr(child_val), ptr(val),
                ptr(self.table)])
        self.iters += B
        self._batches += 1

    def summary(self) -> dict:
        return {**super().summary(), "kernel": "gpu-wavefront", "batch": self.batch,
                "prune": self.prune, "prune_after": self.prune_after,
                "nodes_per_iter": round(self.nodes / max(self.iters, 1), 1)}
