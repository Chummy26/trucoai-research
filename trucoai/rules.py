"""Regulamento configurável de Truco Paulista 1v1.

Cada regra tem um *status* de verificação. Em 19/10/2026 nenhuma regra pôde ser
confirmada no texto publicado da Online (o site não expõe regulamento no
cliente; ver docs/REGRAS.md). Por isso todas começam como ``HIPOTESE`` e devem
ser promovidas a ``OBSERVADO`` ou ``PUBLICADO`` só com evidência registrada.

O ``RuleSet`` é imutável e tem uma impressão digital (``fingerprint``) que deve
ser gravada em todo experimento, para congelar a versão do regulamento usada.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

PUBLICADO = "publicado"  # texto oficial da Online
OBSERVADO = "observado"  # comportamento observado em partidas 1v1 na Online
HIPOTESE = "hipotese"  # convenção comum do Truco Paulista, ainda não verificada


@dataclass(frozen=True)
class RuleSet:
    version: str = "paulista-1v1-h0"
    target_score: int = 12
    # Escada de valores da mão: 1 -> truco(3) -> seis(6) -> nove(9) -> doze(12).
    stake_ladder: Tuple[int, ...] = (1, 3, 6, 9, 12)
    # Não permite pedir aumento que ultrapasse o necessário para fechar a partida
    # quando a aposta atual já basta para isso.
    cap_raise_at_needed: bool = False
    # Mão de onze (só um jogador com 11): ele vê as próprias cartas e decide.
    mao11_value: int = 3
    mao11_run_points: int = 1
    mao11_allows_truco: bool = False
    # Mão de ferro (ambos com 11): valor e se as cartas são jogadas às cegas.
    iron_value: int = 1
    iron_blind: bool = True
    # Carta encoberta: a partir de qual rodada (0-based) pode ser jogada.
    allow_covered: bool = True
    covered_from_round: int = 1
    # Três rodadas empatadas: "nobody" (ninguém pontua) ou "mao" (vence quem é mão).
    all_tied: str = "nobody"
    # Após rodada empatada, quem abre a próxima: "same_leader" ou "mao".
    tie_next_leader: str = "same_leader"
    # Permite "correr" (desistir da mão) na própria vez, sem pedido pendente,
    # até a rodada ``fold_on_turn_max_round`` (0-based).
    allow_fold_on_turn: bool = False
    fold_on_turn_max_round: int = 2
    status: Dict[str, str] = field(default_factory=lambda: {
        "baralho_40_cartas": HIPOTESE,
        "manilha_proxima_da_vira": HIPOTESE,
        "naipes_manilha_paus_copas_espadas_ouros": HIPOTESE,
        "escada_1_3_6_9_12": HIPOTESE,
        "recusa_da_valor_anterior": HIPOTESE,
        "mao_de_onze": HIPOTESE,
        "mao_de_ferro_as_cegas": HIPOTESE,
        "carta_encoberta_a_partir_da_2a_rodada": HIPOTESE,
        "regras_de_empate": HIPOTESE,
        "partida_ate_12": HIPOTESE,
    })

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        d = self.to_dict()
        if not self.allow_fold_on_turn:
            # Campo sem efeito; omitido para manter a impressão digital de h0.
            d.pop("fold_on_turn_max_round")
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def unconfirmed(self) -> list:
        return sorted(k for k, v in self.status.items() if v == HIPOTESE)


DEFAULT_RULES = RuleSet()


# Regulamento com o que foi OBSERVADO na Online em 19/10/2026 (partida 1v1
# Paulista, mesa R$ 0,25; evidência em captures/20261019_dom_partida1/ e
# docs/REGRAS.md). Diferença em relação a h0: "correr" na própria vez existe,
# só na 1ª rodada (flag ``cgr`` do protocolo só é verdadeira antes da 1ª vaza).
# Mão de ferro (19/10/2026, logs/online/20261019_021457_shadow.jsonl): cartas
# visíveis ao próprio jogador, vale 1, sem truco, sem correr e sem virar.
ONLINE_O1 = RuleSet(
    version="paulista-1v1-o1",
    allow_fold_on_turn=True,
    fold_on_turn_max_round=0,
    iron_blind=False,
    status={
        **RuleSet().status,
        "escada_1_3_6_9_12": OBSERVADO,
        "recusa_da_valor_anterior": OBSERVADO,
        "carta_encoberta_a_partir_da_2a_rodada": OBSERVADO,
        "partida_ate_12": OBSERVADO,
        "correr_na_propria_vez": OBSERVADO,
        "sem_limite_de_aumento_perto_de_12": OBSERVADO,
        "mao_de_onze": OBSERVADO,  # correr = +1 ao outro; jogar = vale 3, sem truco
        "regras_de_empate": OBSERVADO,  # parcial: 26 mãos reproduzidas sem divergência
        "manilha_proxima_da_vira": OBSERVADO,
        "naipes_manilha_paus_copas_espadas_ouros": OBSERVADO,
        "mao_de_ferro_as_cegas": OBSERVADO,  # observado: NÃO é às cegas (iron_blind=False)
    },
)
