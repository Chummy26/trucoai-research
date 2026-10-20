"""Compilação (NVRTC) e lançamento (driver CUDA) do kernel, sem toolkit instalado.

Usa o ``nvrtc64_*.dll`` que vem dentro do PyTorch e o ``nvcuda.dll`` do driver.
A memória é alocada com tensores do PyTorch (mesmo contexto primário).
"""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path
from typing import List, Sequence

import torch

KERNEL_SRC = Path(__file__).with_name("mccfr_kernel.cu")
CU_LIMIT_STACK_SIZE = 0


def _nvrtc():
    lib_dir = Path(torch.__file__).parent / "lib"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(lib_dir))
    cands = sorted(glob.glob(str(lib_dir / "nvrtc64_*.dll")))
    cands = [c for c in cands if ".alt." not in c] or cands
    if not cands:
        raise RuntimeError("NVRTC não encontrado no PyTorch")
    return ctypes.CDLL(cands[0])


def _check(res, what):
    if res != 0:
        raise RuntimeError(f"{what} falhou (código {res})")


class Kernel:
    """Módulo CUDA compilado; ``launch(nome, grid, block, args)``."""

    def __init__(self, src_path: Path = KERNEL_SRC, arch: str = None, defines: Sequence[str] = ()):
        torch.cuda.init()
        torch.zeros(1, device="cuda")  # garante o contexto primário ativo nesta thread
        major, minor = torch.cuda.get_device_capability()
        arch = arch or f"sm_{major}{minor}"
        src = Path(src_path).read_text(encoding="utf-8").encode()
        nv = _nvrtc()
        prog = ctypes.c_void_p()
        _check(nv.nvrtcCreateProgram(ctypes.byref(prog), src, b"mccfr_kernel.cu", 0, None, None),
               "nvrtcCreateProgram")
        opts = [f"--gpu-architecture={arch}".encode(), b"--use_fast_math",
                b"-default-device"] + [f"-D{d}".encode() for d in defines]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        res = nv.nvrtcCompileProgram(prog, len(opts), arr)
        logsz = ctypes.c_size_t()
        nv.nvrtcGetProgramLogSize(prog, ctypes.byref(logsz))
        log = ctypes.create_string_buffer(logsz.value)
        nv.nvrtcGetProgramLog(prog, log)
        self.compile_log = log.value.decode(errors="replace")
        if res != 0:
            raise RuntimeError("erro de compilação CUDA:\n" + self.compile_log)
        size = ctypes.c_size_t()
        _check(nv.nvrtcGetCUBINSize(prog, ctypes.byref(size)), "nvrtcGetCUBINSize")
        cubin = ctypes.create_string_buffer(size.value)
        _check(nv.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
        nv.nvrtcDestroyProgram(ctypes.byref(prog))
        self.cu = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        self.module = ctypes.c_void_p()
        _check(self.cu.cuModuleLoadData(ctypes.byref(self.module), cubin), "cuModuleLoadData")
        self._funcs = {}

    def set_stack(self, nbytes: int) -> None:
        _check(self.cu.cuCtxSetLimit(CU_LIMIT_STACK_SIZE, ctypes.c_size_t(nbytes)), "cuCtxSetLimit")

    def func(self, name: str):
        if name not in self._funcs:
            f = ctypes.c_void_p()
            _check(self.cu.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()),
                   f"cuModuleGetFunction({name})")
            self._funcs[name] = f
        return self._funcs[name]

    def launch(self, name: str, grid: int, block: int, args: List) -> None:
        """``args``: lista de ctypes (c_void_p para ponteiros, c_uint64, c_int...)."""
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p)
                                                   for a in args])
        _check(self.cu.cuLaunchKernel(self.func(name), grid, 1, 1, block, 1, 1, 0, None, params, None),
               f"cuLaunchKernel({name})")

    def sync(self) -> None:
        _check(self.cu.cuCtxSynchronize(), "cuCtxSynchronize")


def ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())
