# E13 — seleção de uma política mais equilibrada

20/10/2026. Este documento registra **desenvolvimento e seleção**, não a
confirmação independente. O objetivo mudou de vencer somente o antecessor
para melhorar contra um conjunto fixo de estilos, preservando a observação
legítima do jogador e o regulamento O1 (`ca73a2770a14c415`).

## Treino novo

Professor: E12-96 congelado. 192 partidas completas, sementes
920000–920191, contra Zap v1, heuristic, malandro_rule e E12, com posições
equilibradas. Coleta em aproximadamente 119,4 s, 3614 decisões próprias.
Nenhuma carta privada real do adversário, semente de baralho ou dado humano
foi entrada da rede.

171 partidas de treino, 21 de validação, separadas pelo hash da semente da
partida inteira; 356 decisões na validação. Não houve sobreposição.
Todos os vetores eram finitos e todos os rótulos eram ações legais.
Contagens das nove classes: `[1144, 771, 171, 77, 42, 0, 664, 192, 553]`.
A classe 5 não ocorreu nessa coleta; isso limita a cobertura dos exemplos.

MLP 224–96–96–9, AdamW, taxa 0,001, weight decay 0,001, lotes de 1024,
seleção de época por NLL reservada. Continuação do checkpoint E12 (s71):
NLL 0,86932, acerto 66,29%, melhor época 20. Treino do zero (s72):
NLL 0,99217, melhor época 38. A continuação foi escolhida antes da arena.
Ambos os ajustes levaram cerca de quatro segundos na RTX 3060 local.

Os campos descritivos legados dos scripts podem mencionar mãos independentes
ou divisão por módulo direto. Neste experimento, a unidade real foi a
partida completa e a divisão efetiva foi por hash, registrada em
`runs/e13_development_20261020/selection.json`.

## Matriz de respostas

E9, E11, E12 e a nova política direta enfrentaram Zap v1, heuristic,
malandro_rule, PIMC24, E9, E11 e E12. Sementes 220000–220009,
20 jogos por célula não idêntica, sempre com posições espelhadas.
As três células de uma política contra ela mesma foram fixadas em 50%
por simetria, sem serem tratadas como medições. Total: 500 jogos.

| Política | Média das sete células |
|---|---:|
| E9 | 53,57% |
| E11 | 52,86% |
| E12 | 40,71% |
| Nova rede direta | 56,43% |

Essas médias serviram para desenvolver a seleção. As células pequenas têm
incerteza alta. Não foram usadas como confirmação de superioridade.

A arena antiga concluiu os 500 jogos em 772,3 s, mas a checagem final
rejeitou uma mudança concorrente em `scripts/enable_online_alignment.py`,
um helper de integração que não é importado pela arena. O resultado
original com `completed=false` e o log do erro foram preservados.
`results/e13_matrix_20261020/completion_audit.json` conferiu todos os
placares brutos, pares, conclusões e ações legais, mais os outros 93 hashes.
Somente esse helper não utilizado mudou; os dados foram aceitos apenas
como desenvolvimento. Nenhum trabalho concorrente foi revertido.

## Candidatos e resultados em novas partidas

Sementes 221000–221009, 20 jogos por adversário, sete adversários com
pesos iguais. As alternativas foram:

- Seleção pela probabilidade do estilo e pela matriz de desempenho:
  `PayoffPortfolioAgent`, com suavização dos placares.
- Agrupamento de estilos que compartilham a mesma resposta.
- Busca conjunta sobre mãos hipotéticas e sete modelos de comportamento:
  um estilo permanece fixo durante cada continuação simulada; o histórico
  público atualiza a distribuição de estilos sem contar duas vezes a mesma
  evidência parcial da mão.
- A rede direta recém-treinada.

| Candidato | Média | Pior célula |
|---|---:|---:|
| Seleção por desempenho | 57,14% | 35% |
| Grupos de estilos | 63,57% | 45% |
| **Busca conjunta, n96** | **63,57%** | **55%** |
| Rede direta | 57,14% | 45% |
| Seleção por desempenho com histórico | 63,57% | 45% |
| Grupos com histórico | 64,29% | 45% |
| Busca conjunta, n384 | 57,14% | 45% |
| Controle E9 | 59,29% | 40% |

A busca conjunta n96 fez, por adversário: Zap v1 55%, heuristic 60%,
malandro 70%, PIMC24 65%, E9 70%, E11 60%, E12 65%.
Esses números são seleção; não se devem juntar suas partidas às do teste final.

As versões com histórico reutilizaram resultados anteriores de E9, E11 e
E12 como **dados de desenvolvimento deste novo experimento**. As fontes e
contagens estão em `payoff_history.json`. Células antigas de PIMC64 não
foram agrupadas como se fossem PIMC24. Nenhuma semente do novo teste final
entrou nesse ajuste.

## Escolha e custo

O critério de desenvolvimento foi refinado após a primeira validação:
exigir pelo menos 50% em cada célula e 60% de média, em vez de favorecer
uma média alta com uma fraqueza concentrada. Entre elegíveis, maior média,
depois maior pior célula, depois menor tempo. A alteração foi registrada
em `history_selection_note.md` antes de ler os resultados com histórico.
É seleção guiada por desenvolvimento, não uma hipótese confirmada nesses
mesmos jogos.

**Escolhido: busca conjunta n96/k2**, sete modelos e a rede s71 nas
continuações próprias. Ela foi a única alternativa nova que passou o
filtro de equilíbrio. A versão n384 custou cerca de 1,018 s por decisão
no piloto contra 0,290 s de n96, sem melhorar os resultados observados.
A causa dessa diferença não foi estabelecida; o piloto não prova uma
vantagem universal de usar menos cenários.

A nova arena intercala pares completos entre confrontos, conserva sementes
e resets e verifica os hashes da dependência efetiva dos scripts. Um teste
comparou os resultados com a execução sequencial, inclusive carregando
uma política composta em processos separados. A suíte de fronteiras
passou em 113 testes e o teste adicional da arena passou separadamente.

O congelamento inicial do recibo precisou de uma correção de serialização:
o estado interno de crença não é um parâmetro JSON. A tentativa incompleta
foi preservada em `runs/e13_freeze_incomplete_20261020`; nenhuma partida
confirmatória usou aquela pasta. O congelamento válido é
`runs/e13_frozen_20261020`.

## Confirmação separada

O plano final foi congelado antes de iniciar os jogos:

- Candidato e E9, cada um com 80 partidas contra cada um dos sete
  adversários: 1120 partidas no painel principal, sementes 240000–240039.
- Mais 160 partidas exploratórias, incluindo SimpleX e PIMC64.
- Ganho médio mínimo de 5 pp sobre E9, média do candidato >=60%,
  intervalo pareado para o ganho acima de zero, nenhuma célula abaixo de
  45%, nenhuma ação ilegal e tempos observados abaixo de 10 s.
- Incerteza agregada por semente, mantendo juntas as duas posições e os
  sete estilos. Nenhum aumento de amostra ou troca de candidato após ver
  o resultado final.

Plano e artefatos: `runs/e13_frozen_20261020/plan.json`.
Este documento não declara que os critérios finais foram cumpridos.
Resultados contra bots não demonstram força contra humanos ou integração
validada no fluxo da plataforma de referência.
