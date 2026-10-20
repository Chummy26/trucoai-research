"""Decodificador CBOR mínimo (RFC 8949), sem dependências.

Usado para os anexos binários dos ``gameEvent`` da PlayTruco (socket.io).
Suporta inteiros, bytes, texto, arrays, mapas, tags (ignoradas), booleanos,
null/undefined e floats de 16/32/64 bits, inclusive comprimento indefinido.
"""

from __future__ import annotations

import struct
from typing import Any, Tuple

_BREAK = object()


def loads(data: bytes) -> Any:
    value, pos = _decode(data, 0)
    if pos != len(data):
        raise ValueError(f"sobraram {len(data) - pos} bytes")
    return value


def _arg(data: bytes, pos: int, info: int) -> Tuple[int, int]:
    if info < 24:
        return info, pos
    size = {24: 1, 25: 2, 26: 4, 27: 8}.get(info)
    if size is None:
        raise ValueError(f"informação adicional inválida {info}")
    return int.from_bytes(data[pos:pos + size], "big"), pos + size


def _half(bits: int) -> float:
    s, e, f = bits >> 15, (bits >> 10) & 0x1F, bits & 0x3FF
    if e == 0:
        v = f * 2.0 ** -24
    elif e == 31:
        v = float("nan") if f else float("inf")
    else:
        v = (1 + f / 1024) * 2.0 ** (e - 15)
    return -v if s else v


def _decode(data: bytes, pos: int) -> Tuple[Any, int]:
    ib = data[pos]
    pos += 1
    major, info = ib >> 5, ib & 0x1F
    if major == 7:
        if info == 20:
            return False, pos
        if info == 21:
            return True, pos
        if info in (22, 23):
            return None, pos
        if info == 25:
            return _half(int.from_bytes(data[pos:pos + 2], "big")), pos + 2
        if info == 26:
            return struct.unpack(">f", data[pos:pos + 4])[0], pos + 4
        if info == 27:
            return struct.unpack(">d", data[pos:pos + 8])[0], pos + 8
        if info == 31:
            return _BREAK, pos
        return info, pos  # simple value
    if info == 31 and major in (2, 3, 4, 5):  # comprimento indefinido
        items, mapping = [], {}
        while True:
            v, pos = _decode(data, pos)
            if v is _BREAK:
                break
            if major == 5:
                val, pos = _decode(data, pos)
                mapping[v] = val
            else:
                items.append(v)
        if major == 2:
            return b"".join(items), pos
        if major == 3:
            return "".join(items), pos
        return (mapping if major == 5 else items), pos
    n, pos = _arg(data, pos, info)
    if major == 0:
        return n, pos
    if major == 1:
        return -1 - n, pos
    if major == 2:
        return bytes(data[pos:pos + n]), pos + n
    if major == 3:
        return data[pos:pos + n].decode("utf-8"), pos + n
    if major == 4:
        out = []
        for _ in range(n):
            v, pos = _decode(data, pos)
            out.append(v)
        return out, pos
    if major == 5:
        out = {}
        for _ in range(n):
            k, pos = _decode(data, pos)
            v, pos = _decode(data, pos)
            out[k if not isinstance(k, (list, dict)) else str(k)] = v
        return out, pos
    if major == 6:  # tag: devolve só o conteúdo
        return _decode(data, pos)
    raise ValueError(f"tipo CBOR {major} desconhecido")
