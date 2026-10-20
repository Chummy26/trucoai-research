"""Registro de agentes por especificação textual.

Especificações:
    zap[:v1]  — a IA principal ("Zap"); atalho para a versão registrada em ZAP_VERSIONS
    random | heuristic | pimc[:n] | blueprint:<caminho.pkl>[@min_visitas] | hybrid:<caminho.pkl>[@min_visitas]
    search:<blueprint (pasta compacta ou .pkl)>[?n=48&k=4&belief=1&rollout=blueprint
           &adaptive=0&mv=50&bmv=2000&bets=1&bcards=0&temper=0.5&mix=0]
           (busca com crença bayesiana; agents/search.py)

Referências (adaptadores em ``references/``; recebem só a Observation):
    malandro_rule | malandro_ppo[:<nome ou .zip>] (padrão truco_liga_v2)
    mccfr_simplex | mccfr_simple
"""

from __future__ import annotations

from pathlib import Path

# "Zap" (o 4 de paus, a carta mais forte do Truco Paulista): nome da IA principal.
# Zap v1 = vencedor do E5 (docs/EXPERIMENTOS.md).
ZAP_VERSIONS = {
    "v1": f"search:{Path(__file__).resolve().parents[2] / 'runs' / 'mccfr_rs_o1_s7' / 'compact'}?belief=0&mix=1",
}

from .base import Agent, RandomAgent
from .heuristic import HeuristicAgent
from .pimc import PIMCAgent


def make_agent(spec: str) -> Agent:
    if spec == "zap" or spec.startswith("zap:"):
        version = spec.partition(":")[2] or "v1"
        agent = make_agent(ZAP_VERSIONS[version])
        agent.name = f"Zap {version}"
        agent.meta = {"name": agent.name, "spec": ZAP_VERSIONS[version], **agent.meta}
        return agent
    kind, _, arg = spec.partition(":")
    if kind == "random":
        return RandomAgent()
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "pimc":
        return PIMCAgent(n=int(arg) if arg else 24)
    if kind in ("blueprint", "hybrid"):
        from .blueprint import BlueprintAgent, HybridAgent
        path, _, mv = arg.partition("@")  # hybrid:caminho.pkl@visitas_minimas
        cls = BlueprintAgent if kind == "blueprint" else HybridAgent
        return cls(path, min_visits=float(mv)) if mv else cls(path)
    if kind in ("search", "learned"):
        from .search import SearchAgent
        path, _, query = arg.partition("?")
        opts = dict(x.split("=", 1) for x in query.split("&") if x)
        flag = lambda k, d: opts.get(k, d) in ("1", "true", "sim")  # noqa: E731
        cls, extra = SearchAgent, {}
        if kind == "learned":
            from .learned import LearnedSearchAgent
            cls, extra = LearnedSearchAgent, {"network_path": opts["net"], "neural_bets": flag("nbets", "0")}
        agent = cls(path, **extra, min_visits=float(opts.get("mv", 50)), n=int(opts.get("n", 48)),
                           k=int(opts.get("k", 4)), belief=flag("belief", "1"),
                           rollout=opts.get("rollout", "blueprint"), adaptive=flag("adaptive", "0"),
                           bets_from_blueprint=flag("bets", "1"),
                           bet_min_visits=float(opts.get("bmv", 2000)),
                           belief_cards=flag("bcards", "0"), temper=float(opts.get("temper", 0.5)),
                           opp_mix=float(opts.get("mix", 0.0)))
        if kind == "learned" and (flag("gate", "0") or flag("style", "0")):
            from .learned import AdaptiveLearnedAgent, LearnedSearchAgent, StyleAdaptiveAgent
            general = None
            if "general" in opts:
                general = LearnedSearchAgent(path, opts["general"], neural_bets=True,
                                             n=int(opts.get("n", 48)), k=int(opts.get("k", 4)),
                                             belief=True, belief_cards=True, temper=1,
                                             opp_mix=0, bets_from_blueprint=False)
            if flag("style", "0"):
                if general is None:
                    raise ValueError('style policy requires a general network')
                models = [LearnedSearchAgent(path, p, neural_bets=True, belief_cards=True)
                          for p in opts["style_models"].split(',')]
                return StyleAdaptiveAgent(agent, general, models,
                                          style_n=int(opts.get('style_n', 48)),
                                          threshold=float(opts.get('threshold', 0.75)))
            return AdaptiveLearnedAgent(agent, fallback=general)
        return agent
    if kind in ("malandro_rule", "malandro_ppo", "mccfr_simplex", "mccfr_simple"):
        raise ValueError("Reference adapter omitted from public release; see docs/PROVENANCE.md")
    raise ValueError(f"agente desconhecido: {spec}")
