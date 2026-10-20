r"""Reusable browser launcher for the supported online Paulista 1v1 table.

Configure the destination locally with --url or TRUCOAI_SITE_URL.
Modes: shadow (decide without clicking), play (decide and click), remote
(manual decisions), capture (explicit local recording). Login is manual.
See docs/ONLINE_INTEGRATION.md. No browser profile is distributed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from trucoai.integration.online import capture, run_agent, run_remote, site_url  # noqa: E402


def _parse_stake(raw: Optional[str]) -> Optional[float]:
    """Teto em reais aceito com ou sem "R$" e com vírgula decimal (``2``, ``5,50``)."""
    if raw is None:
        return None
    text = raw.strip().replace("R$", "").replace(" ", "")
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        from trucoai.integration.guard import parse_brl  # aceita "R$ 1.000,00" e variantes
        return parse_brl(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "shadow", "play", "remote"])
    # Default available without historical model artifacts.
    ap.add_argument("--agent", default="heuristic")
    ap.add_argument("--url", default=os.environ.get("TRUCOAI_SITE_URL"),
                    help="endereco da mesa; alternativa: TRUCOAI_SITE_URL")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--screenshots", action="store_true", help="captura imagens locais potencialmente privadas")
    ap.add_argument("--max-hands", type=int, default=None)
    ap.add_argument("--profile", default=str(ROOT / ".pw-profile"))
    ap.add_argument("--cdp", default=None,
                    help="conecta ao seu Chrome aberto com --remote-debugging-port (ex.: http://127.0.0.1:9222)")
    ap.add_argument("--max-stake", default=None, metavar="R$",
                    help="teto da entrada paga em reais (ex.: 2 ou 5,50); sem isto, usa TRUCOAI_MAX_STAKE; "
                         "sem nenhum dos dois, a IA só joga mesas grátis")
    a = ap.parse_args()
    if a.url:
        os.environ["TRUCOAI_SITE_URL"] = a.url
    try:
        site_url()
    except ValueError as exc:
        ap.error(str(exc))
    max_stake = _parse_stake(a.max_stake)
    if a.max_stake and max_stake is None:
        ap.error(f"--max-stake {a.max_stake!r} não é um valor em reais legível (ex.: 2 ou 5,50)")
    if a.mode == "capture":
        out = ROOT / "captures" / time.strftime("%Y%m%d_%H%M%S")
        capture(out, Path(a.profile), a.minutes, screenshots=a.screenshots, cdp=a.cdp)
        print(f"captura salva em {out}")
    elif a.mode == "remote":
        run_remote(Path(a.profile), ROOT / "logs" / "online", cdp=a.cdp)
    else:
        run_agent(a.mode, a.agent, Path(a.profile), ROOT / "logs" / "online", cdp=a.cdp,
                  max_hands=a.max_hands, max_stake=max_stake)


if __name__ == "__main__":
    main()
