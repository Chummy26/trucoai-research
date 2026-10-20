"""Converte uma captura da Online em eventos decodificados e anonimizados.

Remove apelidos, IDs de usuário/sala/sessão, skins e URLs assinadas. A saída
pode ir para tests/data/ (versionada); a captura bruta continua fora do git.

Uso:
    python scripts/sanitize_capture.py captures/20261019_014228 tests/data/online_partida2.jsonl
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trucoai.integration.cbor import loads  # noqa: E402

DROP = {"uid", "usn", "sk", "sid", "rid", "iu", "ts", "uids"}


def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items() if k not in DROP}
    if isinstance(o, list):
        return [scrub(v) for v in o]
    return o


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    out = []
    for line in open(src / "frames.jsonl", encoding="utf-8"):
        r = json.loads(line)
        d = r["data"]
        if r["dir"] == "recv" and "b64" in d:
            ev = loads(base64.b64decode(d["b64"]))
            if isinstance(ev, dict) and "tp" in ev:
                out.append({"t": r["t"], "dir": "recv", "ev": scrub(ev)})
        elif r["dir"] == "recv" and isinstance(d.get("json"), list) and d["json"][:1] == ["startGame"]:
            out.append({"t": r["t"], "dir": "recv", "msg": ["startGame"]})
        elif r["dir"] == "sent" and isinstance(d.get("json"), list):
            out.append({"t": r["t"], "dir": "sent", "msg": scrub(d["json"])})
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(out)} registros -> {dst}")


if __name__ == "__main__":
    main()
