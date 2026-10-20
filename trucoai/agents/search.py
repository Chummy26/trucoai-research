"""Busca com crença bayesiana e rollouts guiados pelo blueprint ("PIMC+").

Melhorias sobre ``PIMCAgent`` (cada uma pode ser ligada/desligada para medir):

1. **Crença bayesiana sobre a mão adversária** (``belief``): sorteia ``k·n``
   mundos compatíveis com a observação e reamostra ``n`` com peso igual à
   verossimilhança das ações que o adversário JÁ tomou nesta mão, segundo o
   blueprint (modelo de equilíbrio do adversário). Ex.: quem pediu truco tem
   mais chance de ter manilha; quem jogou encoberta, de ter carta fraca. O PIMC
   original sorteava mãos uniformemente, ignorando essa informação.
2. **Rollouts pelo blueprint** (``rollout="blueprint"``): depois da ação
   avaliada, os dois lados jogam pela política média do MCCFR (inclusive pedir,
   aceitar e correr), e não por uma regra gulosa sem apostas.
3. **Números aleatórios comuns**: o mesmo gerador por mundo para todas as
   ações candidatas, reduzindo a variância da comparação entre ações.
4. **Modelo adaptativo do adversário** (``adaptive``): conta, na partida, quantas
   vezes ele correu de pedidos e corrige a probabilidade de corrida nos rollouts
   (média bayesiana com o prior do blueprint). Explora quem corre demais ou de menos.
5. Apostas pelo blueprint quando o nó é bem treinado (``bets_from_blueprint``),
   como no ``HybridAgent``.

O agente só recebe ``Observation``; mundos vêm de ``sample_state``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ..cfr.abstraction import info_key, key_fn
from ..cfr.mccfr import abstract_legal, to_abstract, to_concrete
from ..engine import ACCEPT, COVER_0, PLAY_0, RAISE, RUN, GameState
from ..equity import default_table
from ..observation import Observation, observe, sample_state
from .base import Agent
from .blueprint import BET_ACTIONS, load_policy
from .pimc import _respond_model
from .rollout import greedy_card

EPS = 0.15  # suavização da verossimilhança: o adversário pode desviar do equilíbrio


class OpponentStats:
    """Frequência de corrida do adversário diante de pedidos, na partida atual."""

    def __init__(self, prior_strength: float = 6.0):
        self.prior_strength = prior_strength
        self.responses = 0
        self.runs = 0
        self.expected_runs = 0.0  # soma das prob. de corrida do blueprint nesses pontos

    def factor(self) -> float:
        """Multiplicador da probabilidade de corrida (1 = como o blueprint)."""
        if self.responses == 0:
            return 1.0
        a = self.prior_strength
        exp_rate = (self.expected_runs + 0.5) / (self.responses + 1.0)
        obs_rate = (self.runs + a * exp_rate) / (self.responses + a)
        return max(0.25, min(4.0, obs_rate / max(exp_rate, 1e-3)))


class SearchAgent(Agent):
    name = "search"

    def __init__(self, policy_path, min_visits: float = 50.0, n: int = 48, k: int = 4,
                 belief: bool = True, rollout: str = "blueprint", adaptive: bool = False,
                 bets_from_blueprint: bool = True, bet_min_visits: float = 2000.0,
                 belief_cards: bool = False, temper: float = 0.5, opp_mix: float = 0.0):
        super().__init__()
        self.store = load_policy(policy_path)
        self.key = key_fn(getattr(self.store, "meta", {}).get("abstraction", "v1"))
        self.meta = {"policy": str(policy_path), **getattr(self.store, "meta", {})}
        self.min_visits = min_visits
        self.bet_min_visits = bet_min_visits
        self.n, self.k = n, k
        self.belief, self.rollout_kind, self.adaptive = belief, rollout, adaptive
        self.bets_from_blueprint = bets_from_blueprint
        # Crença: por padrão só as APOSTAS do adversário entram na verossimilhança
        # (humanos e outros bots escolhem cartas por critérios que o blueprint não
        # modela bem); ``temper`` < 1 suaviza os pesos (evita colapsar em poucos mundos).
        self.belief_cards, self.temper = belief_cards, temper
        # Robustez: nos rollouts, a fração ``opp_mix`` das decisões de APOSTA do
        # adversário segue o modelo de equidade (e não o blueprint). Evita que a
        # busca só funcione contra adversários parecidos com o próprio blueprint.
        self.opp_mix = opp_mix
        self.respond = _respond_model(4)
        self.stats = OpponentStats()
        self._seen_log = 0
        self._pending_raise: Optional[int] = None  # mão em que pedimos e ainda não vimos a resposta

    def reset(self, seed=None):
        super().reset(seed)
        self.stats = OpponentStats()
        self._seen_log = 0
        self._pending_raise = None

    # ------------------------------------------------------------------ blueprint
    def _dist(self, obs: Observation, min_visits: float) -> Optional[Dict[int, float]]:
        dist, visits = self.store.lookup(self.key(obs))
        return dist if dist is not None and visits >= min_visits else None

    def _policy_probs(self, obs: Observation) -> Optional[Dict[int, float]]:
        """Probabilidades sobre ações CONCRETAS legais (None se nó não treinado)."""
        dist = self._dist(obs, self.min_visits)
        if dist is None:
            return None
        out = {}
        for a in obs.legal_actions:
            out[a] = dist.get(to_abstract(obs, a), 0.0)
        tot = sum(out.values())
        if tot <= 0:
            return None
        return {a: p / tot for a, p in out.items()}

    # ------------------------------------------------------------------ crença
    def _likelihood(self, world: GameState, obs: Observation) -> float:
        """P(ações do adversário nesta mão | mãos do mundo), pelo blueprint."""
        me, opp = obs.me, obs.opp
        start = _hand_start(world, obs)
        if start is None:
            return 1.0
        s, plays = start
        lik, pi = 1.0, 0
        for t, p, v in obs.public_log[1:]:
            if t in ("round", "hand_end"):
                continue
            if t == "play":
                pl, card, cov = plays[pi]
                pi += 1
                slot = s.hands[pl].index(card)
                a = (COVER_0 if cov else PLAY_0) + slot
            elif t == "raise":
                a = RAISE
            elif t == "accept":
                a = ACCEPT
            elif t in ("run", "fold"):
                a = RUN
            elif t == "mao11":
                a = v
            else:
                return lik
            if s.current == opp and (self.belief_cards or t != "play"):
                o = observe(s, opp)
                probs = self._policy_probs(o)
                legal = o.legal_actions
                p_bp = probs.get(a, 0.0) if probs else 1.0 / len(legal)
                lik *= (1 - EPS) * p_bp + EPS / len(legal)
            if a not in s.legal_actions():
                return lik  # mundo inconsistente com o log (não deveria ocorrer)
            s.apply(a)
        return lik

    def worlds(self, obs: Observation) -> List[GameState]:
        if not self.belief or not any(p == obs.opp for _, p, _ in obs.public_log[1:]):
            return [sample_state(obs, self.rng) for _ in range(self.n)]
        cand = [sample_state(obs, self.rng) for _ in range(self.n * self.k)]
        w = [self._likelihood(c, obs) ** self.temper for c in cand]
        if sum(w) <= 0:
            return cand[: self.n]
        return self.rng.choices(cand, weights=w, k=self.n)

    # ------------------------------------------------------------------ rollout
    def _rollout_action(self, s: GameState, rng: random.Random, me: int) -> int:
        p = s.current
        if p != me and self.opp_mix > 0 and s.phase in ("respond", "mao11") and rng.random() < self.opp_mix:
            return self.respond(s, rng) if s.phase == "respond" else ACCEPT
        if self.rollout_kind == "blueprint":
            o = observe(s, p)
            probs = self._policy_probs(o)
            if probs:
                if self.adaptive and p != me and s.phase == "respond" and RUN in probs:
                    f = self.stats.factor()
                    probs = dict(probs)
                    probs[RUN] = min(0.999, probs[RUN] * f)
                    rest = sum(v for a, v in probs.items() if a != RUN)
                    if rest > 0:
                        scale = (1 - probs[RUN]) / rest
                        probs = {a: (v if a == RUN else v * scale) for a, v in probs.items()}
                acts = list(probs)
                return rng.choices(acts, weights=[probs[a] for a in acts])[0]
        if s.phase == "play":
            return greedy_card(s, rng)
        if s.phase == "respond":
            return self.respond(s, rng)
        return ACCEPT

    def _rollout(self, s: GameState, rng: random.Random, me: int) -> float:
        while not s.is_terminal():
            s.apply(self._rollout_action(s, rng, me))
        return default_table()(s.scores[me], s.scores[1 - me])

    def action_values(self, obs: Observation, actions=None) -> Dict[int, float]:
        actions = list(actions if actions is not None else obs.legal_actions)
        vals = {a: 0.0 for a in actions}
        ws = self.worlds(obs)
        for w in ws:
            seed = self.rng.getrandbits(32)
            for a in actions:
                s = w.clone()
                s.apply(a)
                vals[a] += self._rollout(s, random.Random(seed), obs.me)
        return {a: v / len(ws) for a, v in vals.items()}

    # ------------------------------------------------------------------ modelo do adversário
    def _count(self, ran: bool) -> None:
        self.stats.responses += 1
        self.stats.runs += ran
        self.stats.expected_runs += 0.3  # taxa de corrida a priori (aprox. do blueprint)

    def _update_stats(self, obs: Observation) -> None:
        """Conta respostas do adversário a pedidos nossos desde a última decisão.

        Se a mão em que pedimos acabou sem voltarmos a jogar, o adversário correu
        (após aceitar, a vez volta a quem pediu).
        """
        if self._pending_raise is not None and self._pending_raise != obs.hand_no:
            self._count(True)
            self._pending_raise = None
            self._seen_log = 0
        log = obs.public_log
        if len(log) < self._seen_log:
            self._seen_log = 0
        for i in range(max(self._seen_log, 1), len(log)):
            t, p, _ = log[i]
            prev = log[i - 1]
            if p == obs.opp and t in ("accept", "run", "raise") and prev[0] == "raise" and prev[1] == obs.me:
                self._count(t == "run")
                self._pending_raise = None
        self._seen_log = len(log)

    # ------------------------------------------------------------------ decisão
    def act(self, obs: Observation) -> int:
        if self.adaptive:
            self._update_stats(obs)
        a = self._decide(obs)
        if a == RAISE:
            self._pending_raise = obs.hand_no
        return a

    def _decide(self, obs: Observation) -> int:
        legal = obs.legal_actions
        if len(legal) == 1:
            return legal[0]
        if self.bets_from_blueprint:
            dist = self._dist(obs, self.bet_min_visits)
            if dist is not None:
                acts = list(dist)
                ab = self.rng.choices(acts, weights=[dist[x] for x in acts])[0]
                a = to_concrete(obs, ab)
                if a in legal and (a in BET_ACTIONS or obs.phase != "play"):
                    return a
                if a in legal:  # blueprint não quer apostar: carta pela busca
                    cand = [x for x in legal if x not in (RAISE, RUN)]
                    vals = self.action_values(obs, cand)
                    return max(vals, key=vals.get)
        vals = self.action_values(obs)
        return max(vals, key=vals.get)


def _hand_start(world: GameState, obs: Observation) -> Optional[Tuple[GameState, list]]:
    """Reconstrói o estado do início da mão com as cartas do mundo sorteado."""
    plays = [(p, c, cov) for rnd in world.rounds for (p, c, cov) in rnd]
    full = {0: [], 1: []}
    for p in (0, 1):
        full[p] = [c for c in world.hands[p] if c is not None] + [c for (q, c, _) in plays if q == p]
    if len(full[0]) != 3 or len(full[1]) != 3:
        return None
    s = GameState(rules=obs.rules, dealer=obs.dealer)
    s.scores = list(obs.scores)
    mao = 1 - obs.dealer
    deck = []
    for i in range(3):
        deck += [full[mao][i], full[1 - mao][i]]
    used = set(deck) | {world.vira}
    deck += [world.vira] + [c for c in range(40) if c not in used]
    s.deal(random.Random(0), deck)
    return s, plays
