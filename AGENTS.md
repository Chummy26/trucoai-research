# Orientações para agentes

## Objetivo do projeto

Este projeto é uma pesquisa sobre inteligência artificial aplicada ao **Truco Paulista 1v1**, com dois jogadores individuais, sem duplas. O objetivo é desenvolver, treinar e avaliar uma IA capaz de jogar contra humanos, respeitando as regras e a informação disponível a cada jogador.

A referência de regras escolhida pelo usuário é a modalidade Truco Paulista da [PlayTruco](https://www.playtruco.com/truco). O projeto deve reproduzir as regras dessa modalidade no formato 1v1.

O ambiente alvo de atuação da IA é o próprio site da **PlayTruco**: a IA será treinada para disputar partidas de Truco Paulista 1v1 contra jogadores humanos nessa plataforma. A pesquisa inclui a integração do modelo com o fluxo real de partida do site. Simuladores locais e partidas de autojogo servem ao treinamento e à validação; o objetivo final é jogar contra pessoas na PlayTruco.

## Princípios de trabalho

- Trate o projeto como pesquisa: formule hipóteses verificáveis, registre experimentos e apresente evidências para as conclusões.
- Preserve alterações e trabalhos existentes. Antes de editar, examine os arquivos e o estado do repositório, quando houver controle de versão.
- Faça mudanças pequenas, rastreáveis e coerentes com o objetivo do projeto.
- A variante e o formato já estão definidos: Truco Paulista 1v1 conforme a PlayTruco. Não substitua por Truco Mineiro, partidas em duplas ou outra convenção de regras. Para detalhes ainda não confirmados, linguagem, arquitetura ou infraestrutura, verifique decisões já documentadas e esclareça ambiguidades que alterem o jogo ou a pesquisa.
- Não apresente resultados de simulação como evidência de desempenho contra humanos.

## Regras e ambiente de jogo

- Documente o regulamento de Truco Paulista 1v1 da PlayTruco: baralho, distribuição e vira, hierarquia das cartas, manilhas, ordem de jogo, pontuação, empates, pedidos e respostas, recusas, mão de onze, eventual mão de ferro e cartas encobertas.
- Confira cada detalhe na referência antes de implementá-lo como regra confirmada. Não complete lacunas com regras genéricas de Truco Paulista sem registrar e validar a diferença.
- Registre a origem e a data de verificação das regras, distinguindo texto publicado, comportamento observado na modalidade 1v1 e hipóteses ainda não confirmadas. Congele uma versão do regulamento para cada experimento.
- Estado da consulta em 19/10/2026: a página indicada apresentou a interface de partida, mas não forneceu regulamento textual extraível na consulta realizada. A variante, o formato e a referência estão definidos pelo usuário; os detalhes do regulamento ainda precisam ser verificados.
- Mantenha as regras e a validação de ações independentes da política da IA e da interface.
- O ambiente deve determinar as ações legais, a transição de estado e o resultado de cada partida.
- Cubra com testes os casos relevantes das regras, incluindo encerramento de rodadas e partidas, empates e aumentos de aposta.

## Informação disponível à IA

- Separe o estado completo do jogo da observação entregue a cada jogador.
- Na inferência, a IA só pode acessar informações permitidas pelas regras: suas cartas, eventos públicos e demais observações legítimas.
- Não exponha cartas ocultas, ordem futura do baralho, sementes do embaralhamento ou estados privados dos adversários à política que decide as jogadas.
- Se o treinamento usar informação privilegiada, documente esse uso e verifique que ela não é necessária nem acessível durante partidas contra humanos.
- Garanta as mesmas regras e os mesmos limites de observação para IA e humanos; registre eventuais diferenças de tempo ou recursos de decisão.

## Treinamento e reprodutibilidade

- Registre versão das regras e do código, origem dos dados, configurações, sementes, orçamento computacional e checkpoints dos experimentos.
- Separe dados de treino, validação e teste, evitando vazamento entre partidas e versões de políticas. Quando aplicável, separe também jogadores humanos.
- Mantenha checkpoints e resultados identificáveis; não sobrescreva artefatos de experimentos já reportados.
- Use adversários de referência e versões anteriores da política para medir progresso. Bom desempenho em autojogo, isoladamente, não comprova força contra humanos.
- Diferencie resultados reproduzíveis, observações preliminares e hipóteses ainda não testadas.

## Avaliação

- Declare protocolo, adversários, número de partidas, distribuição de cartas, posições e orçamento de decisão.
- Meça desempenho em partidas e pontuação, frequência de ações ilegais e tempo de decisão. Relate incerteza e limitações das métricas.
- Equilibre posições e distribuições de cartas nas comparações, quando compatível com o protocolo.
- Separe evidências de testes de regras, simulações, autojogo, avaliação contra bots e partidas com humanos.
- Verifique o modelo efetivamente carregado pelo jogo antes de atribuir resultados a um checkpoint.

## Partidas com humanos e dados

- Colete apenas os dados necessários, evite identificadores pessoais nos registros e não versione credenciais ou dados pessoais.
- Documente finalidade, consentimento, retenção e exclusão dos dados coletados antes de iniciar uma coleta com participantes.
- Mantenha registros suficientes para investigar erros e reproduzir partidas, sem expor informações privadas aos jogadores durante o jogo.

## Entrega e comunicação

- Ao concluir uma tarefa, descreva o que mudou, como foi verificado e quais limitações permanecem.
- Execute verificações proporcionais à mudança. Alterações nas regras, observações ou integração do modelo exigem testes dessas fronteiras.
- Não declare o sistema pronto para jogar contra humanos apenas porque o treinamento ou os testes isolados passaram; verifique também a integração no fluxo real de partida.
