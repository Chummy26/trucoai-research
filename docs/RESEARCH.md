# Relatório de pesquisa — E13

Data dos registros: 20/10/2026. Estudo computacional exploratório, sem revisão por pares. Objetivo: melhorar a média de vitória contra um painel previamente fixado de bots, usando somente a observação legítima no momento de decidir.

## Método

Variante: Paulista 1v1, versão `paulista-1v1-o1`, fingerprint `ca73a2770a14c415`. O1 é um modelo experimental baseado em observações históricas; detalhes não observados permanecem hipóteses em REGRAS.md.

A política E13 usa busca conjunta sobre mãos hipotéticas e modelos comportamentais, com n=96 e k=2. Uma rede 224–96–96–9 orienta continuações próprias. Treino relatado: 192 partidas sintéticas completas do professor E12-96; 171 partidas de treino e 21 de validação, separadas por hash da partida. As sementes da confirmação não entraram na seleção. A classe de ação 5 não ocorreu no conjunto de treino. Configurações e seleção histórica estão em E13_DEVELOPMENT.md.

Confirmação: sementes 240000–240039, duas posições por semente e 80 partidas por adversário, para E13 e E9. Sete famílias com o mesmo peso; 1.120 partidas principais. Outras 160 partidas são exploratórias e não entram no ganho principal. Orçamento declarado: até 10 s por decisão. Execução histórica: 12 processos e aproximadamente 759,8 s; não é um limite garantido para outras máquinas.

Unidade de incerteza: semente pareada, mantendo juntas as duas posições e as sete famílias. O intervalo do ganho usa a diferença entre candidatos para as mesmas sementes, média e erro padrão entre 40 blocos, com aproximação normal de 1,96. As famílias são fixas; o intervalo não cobre generalização para novos adversários. Seleção anterior do candidato também limita inferências amplas.

## Resultados principais

| Adversário | Vitórias E13 | Taxa E13 | Taxa E9 |
|---|---:|---:|---:|
| v1 | 38/80 | 47.50% | 50.00% |
| heuristic | 60/80 | 75.00% | 71.25% |
| malandro | 64/80 | 80.00% | 78.75% |
| pimc | 52/80 | 65.00% | 57.50% |
| e9 | 52/80 | 65.00% | 55.00% |
| e11 | 47/80 | 58.75% | 40.00% |
| e12 | 50/80 | 62.50% | 46.25% |

Total: E13 363/560 (64,82%); E9 319/560 (56,96%). Ganho: 7,86 pp, IC95% [2,02; 13,70] pp. Nenhuma ação ilegal foi registrada para candidato, controle ou adversários nas 1.280 partidas.

E13: 12.740 decisões no painel principal, média 0,302 s, p99 0,572 s, máximo 0,938 s. O teste não demonstra superioridade em cada família; contra Zap v1 a estimativa foi 47,5%. Não há avaliação humana neste relatório.

## Evidência e reprodução

Os recibos exportados estão em `research/e13/`. `python scripts/audit_e13.py` verifica a integridade da exportação e recalcula contagens e intervalo pareado. Esse comando audita as estatísticas armazenadas; não executa novamente a arena nem certifica os 148 hashes das dependências originais.

O registro original relata carregamento verificado na arena. Isso não equivale a validar um loader de entrega, os pesos desta exportação ou uma sessão atual do site. O loader local de entrega estava incompleto e não foi distribuído como funcional.

Blue­prints históricos, configurações dependentes desses artefatos e adaptadores de referências sem licença identificada não são distribuídos. Por isso a confirmação E13 completa não é reproduzível somente com este pacote. Os pesos próprios incluídos não resolvem essa ausência. Não substitua silenciosamente os bots ausentes para alegar reprodução.

## Próximas verificações

Obter autorização/licença das dependências omitidas; disponibilizar configuração portátil e artefatos identificados; repetir confirmação com dependências disponíveis; validar fronteira de observação no fluxo completo; e definir protocolo com consentimento antes de qualquer avaliação humana.
