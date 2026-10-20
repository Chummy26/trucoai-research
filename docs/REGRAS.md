# Regulamento — Truco Paulista 1v1 (referência: PlayTruco)

Versão congelada no código: `RuleSet(version="paulista-1v1-h0")` em
[`trucoai/rules.py`](../trucoai/rules.py). Todo checkpoint e todo resultado grava
`rules_fingerprint` (hash do regulamento).

## Estado da verificação (19/10/2026)

| Fonte | Resultado |
|---|---|
| `https://www.playtruco.com/truco` via fetch | HTTP 403 |
| Mesma página no navegador | Só a interface da mesa ("Partida / Rodadas / Tentos / Pontos"); nenhum texto de regulamento |
| Página inicial | Lobby de **mesas ranqueadas com valores em R$** (R$ 0,25 a R$ 200, com taxa), "Truco Paulista e mineiro com premiação na hora" |
| Bundles JavaScript do cliente | Nenhum texto sobre mão de onze, mão de ferro, encoberta ou manilhas: as regras rodam no servidor |

**Conclusão (consulta inicial):** nenhuma regra PUBLICADA. Parte delas foi depois
OBSERVADA numa partida real (seção seguinte). Para promover uma regra é
preciso registrar a evidência (captura de tela, gravação de partida ou texto
oficial, com data) nesta página.

## Evidências observadas — partida 1 (19/10/2026)

Partida 1v1 Paulista em mesa de R$ 0,25 jogada pelo usuário, gravada pelo texto
da página (`captures/20261019_dom_partida1/`, fora do git). Regulamento derivado:
`PLAYTRUCO_O1` em `trucoai/rules.py` (o padrão `h0` foi mantido porque o núcleo
Rust e os checkpoints existentes o fixam).

| Regra | Evidência (t em ms) | Status |
|---|---|---|
| Escada 1→3→6→9→12 | "Tentos" 1→3→6→9→12 com "pediu Truco/Seis/Nove/Doze" (149113–154214) | OBSERVADO |
| Recusar dá o valor anterior | truco do oponente recusado: 6×2 → 6×3, "Tô Fora!" (84283) | OBSERVADO (só no nível 1→3) |
| Correr na própria vez, sem pedido | botões "Correr / Truco!" na vez; "Você correu!" 6×1 → 6×2 (74264) | OBSERVADO — **diverge de h0** |
| Encoberta a partir da 2ª rodada | botão "Virar" só após a 1ª rodada (39044, 49484, …) | OBSERVADO (efeito na disputa não observado) |
| Responder com aumento | "Recusar / Aceitar / Seis!", "… / Nove!", "… / Doze!" | OBSERVADO |
| Sem limite de aumento perto de 12 | nove e doze pedidos com 8×10 (151916–152934) | OBSERVADO |
| Partida até 12 | "Você perdeu o jogo" em 8×12 (156254) | OBSERVADO |
| Valor da mesa na tela | "R$ 0.5" (prêmio) durante a partida; lobby "R$ 0.25 / Taxa: 0.03" | OBSERVADO (usado pela trava) |
| Mão de onze, mão de ferro, empates, 3 empates | não ocorreram nesta partida | ver protocolo abaixo |

## Evidências pelo protocolo — capturas 2 e 3 (19/10/2026)

Mensagens reais da mesa (`tests/data/playtruco_partida2.jsonl`, `..._partida3.jsonl`,
anonimizadas) reproduzidas no motor com `PLAYTRUCO_O1`: **26 mãos, 0 divergências**
(`tests/test_playtruco_protocol.py`). Isso confirma, nessas mãos, força das cartas,
manilhas e naipes, decisão das rodadas e da mão, pontos de truco/recusa, além de:

| Regra | Evidência | Status |
|---|---|---|
| Ordem das figuras | vira de código 11 fez manilha a carta 12 (13♦ perdeu para 3♠): força por código 4 5 6 7 11 12 13 A 2 3 | OBSERVADO (nome exibido de 11/12 não conferido) |
| Correr na vez só na 1ª rodada | flag `cgr` só verdadeira antes da 1ª vaza; `finish_round gu=true` valendo o valor atual | OBSERVADO |
| Encoberta perde para carta aberta | encobertas do adversário reproduzidas sem divergência | OBSERVADO (duas encobertas: não ocorreu) |
| Aumento em resposta até 12 | `reraiseIncreasePoints` 3→6→9→12 | OBSERVADO |
| Mão de onze: correr | `last_point` e fim da mão sem jogadas: 5×11 → 6×11 | OBSERVADO |
| Mão de onze: jogar | `run_turn crp=3`, `catip=false` (sem truco) | OBSERVADO |
| Placar final limitado a 12 | 11 + 3 exibido como 12 | OBSERVADO |
| Mão de ferro, três empates | não ocorreram nestas capturas | ver abaixo |

## Evidências do primeiro `shadow` ao vivo (19/10/2026, 02:14)

`logs/playtruco/20261019_021457_shadow.jsonl`: partida de R$ 0,25 jogada pelo
usuário, 16 mãos lidas ao vivo, 0 divergências, 12×11 para o usuário.

| Regra | Evidência | Status |
|---|---|---|
| Mão de onze: correr dá 1 ao outro | adversário com 11 correu 10 mãos seguidas (1×11 → 11×11) | OBSERVADO |
| Mão de ferro **não** é às cegas | em 11×11 o `start_round` trouxe as cartas do jogador visíveis | OBSERVADO — **diverge de h0** (`iron_blind=False` em `PLAYTRUCO_O1`) |
| Mão de ferro: vale 1, sem truco, sem correr, sem virar | `run_turn` com `crp=1`, `catip/cgr/chc = false`; venceu 11 → 12 | OBSERVADO |
| Sem correr na vez em mão de onze/ferro | `cgr=false` nas duas | OBSERVADO |
| Três empates | não ocorreu | HIPÓTESE |

## Regras implementadas (h0: todas HIPÓTESE, configuráveis)

| Item | Implementação padrão | Parâmetro |
|---|---|---|
| Baralho | 40 cartas (sem 8, 9, 10 e coringas) | — |
| Distribuição | 3 cartas alternadas a partir do "mão"; depois a vira | — |
| Força comum | 3 > 2 > A > K > J > Q > 7 > 6 > 5 > 4; mesmo valor empata | — |
| Manilha | valor seguinte ao da vira (ciclo 4…3→4) | — |
| Naipes da manilha | Paus (zap) > Copas > Espadas > Ouros | — |
| Quem abre | quem não deu as cartas; o vencedor da rodada abre a seguinte | — |
| Empate de rodada | abre de novo quem abriu a rodada empatada | `tie_next_leader` |
| Decisão da mão | 2 rodadas; 1ª empatada → próxima decidida vence; 1ª decidida + empate depois → vence quem ganhou a 1ª | — |
| Três empates | ninguém pontua | `all_tied` (`"nobody"`/`"mao"`) |
| Apostas | 1 → truco 3 → seis 6 → nove 9 → doze 12 | `stake_ladder` |
| Pedido | só na própria vez; quem fez o último aumento aceito não pede de novo | — |
| Resposta | aceitar, correr (o outro ganha o valor anterior ao pedido) ou aumentar | — |
| Limite ao necessário | desligado | `cap_raise_at_needed` |
| Correr sem pedido | não permitido | `allow_fold_on_turn` |
| Carta encoberta | a partir da 2ª rodada; perde para qualquer carta aberta; duas encobertas empatam; o adversário nunca vê a carta | `allow_covered`, `covered_from_round` |
| Mão de onze | só quem tem 11 decide: jogar (vale 3) ou correr (adversário +1); sem truco | `mao11_*` |
| Mão de ferro | ambos com 11: sem truco, **cartas às cegas para os dois** (nem o próprio jogador vê), vale 1 | `iron_*` |
| Fim | 12 pontos | `target_score` |

Diferença deliberada em relação a `truco-de-malandro`: lá a IA via as próprias
cartas na mão de ferro enquanto o humano não ("regra da casa"). Aqui as duas
partes têm a mesma restrição, como exige `AGENTS.md`.

## O que precisa ser verificado na PlayTruco (modo 1v1 Paulista)

1. Existe carta encoberta? Em que rodadas? O que acontece com duas encobertas?
2. Mão de onze: valor (3?), pontos ao correr (1?), o que acontece se o adversário pedir truco.
3. Mão de ferro: às cegas? valor?
4. Três empates: ninguém pontua ou vence o mão?
5. Após empate, quem abre?
6. "Correr" sem pedido pendente existe no 1v1?
7. Limite de aumento perto de 12 pontos.
8. Tempo por jogada (para medir orçamento de decisão comparável).

Procedimento sugerido: jogar partidas **gratuitas/de treino** (se houver) ou ler a
ajuda oficial; registrar cada caso com data, captura e versão do site.
