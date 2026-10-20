# Arquitetura

```
trucoai/
  rules.py        RuleSet imutável, status de verificação, fingerprint
  cards.py        cartas 0..39, força relativa à vira
  engine.py       GameState: ações legais, transições, pontuação (sem política)
  observation.py  observe(state, p) — única ponte para políticas; sample_state(obs)
  equity.py       P(vencer a partida | placar)
  cfr/            abstração de infosets + MCCFR (ES, RM+, média linear); compact.py (blueprint em numpy)
  agents/         random, heuristic, pimc, blueprint, hybrid, search (crença + rollouts do blueprint)
  agents/references/  adaptadores dos agentes de referência (malandro: RulePlayer e PPO em numpy;
                  mccfr_simplex); recebem só Observation (docs/EXPERIMENTOS.md, E4)
  arena.py        partidas espelhadas, Wilson, métricas
scripts/          train_mccfr, train_mccfr_rs, evaluate, compare_references, estimate_equity, play_cli
tests/            regras, fronteira de informação, agentes
```

## Fronteiras

1. **Regras ↔ política.** `engine.py` não conhece agentes. Agentes recebem só
   `Observation`. A arena chama `observe()` a cada decisão.
2. **Informação.** `observe()` omite cartas do adversário, cartas encobertas dele,
   baralho e sementes. Na mão de ferro às cegas, omite também as próprias cartas.
   `tests/test_observation.py` embaralha toda informação oculta e exige
   observação e chave de infoset idênticas.
3. **Busca sem trapaça.** PIMC e o estimador de força sorteiam mundos com
   `sample_state(obs)` — funcionam igual a partir de uma interface externa.
4. **Treino com informação privilegiada.** O MCCFR percorre estados completos
   (necessário para amostragem externa), mas os nós são indexados só por
   `info_key(observe(...))`. A política resultante depende só da observação.

## Algoritmos

### Utilidade = equidade da partida
`EquityTable` resolve por programação dinâmica `W(a,b)` assumindo que cada mão é
50/50 com distribuição de pontos empírica (`data/equity_dist.json`). Hipóteses:
ignora vantagem de ser mão e decisões especiais da mão de onze.

### Blueprint: MCCFR
- Jogo = uma mão; raiz sorteia placar (50% uniforme, 35% início, 15% com alguém em 11).
- Amostragem externa; RM+; média linear (peso t).
- Ações abstratas: jogar/cobrir a k-ésima carta mais fraca, truco, aceitar, correr.
- Abstração: 8 faixas de força, histórico da mão com faixas das cartas abertas,
  resultado das rodadas, apostas, faixa de placar (≥9) e tipo de mão.
- Paralelismo: processos independentes somados no fim (aproximação: média de
  execuções independentes, não CFR sincronizado).
- Poda de nós raros para caber em RAM (limitação desta máquina: ~2–3 GB livres).
- **Núcleo Rust** (`rust/mccfr`, `scripts/train_mccfr_rs.py`): porta do motor,
  da chave `info_key` e do ES-MCCFR, sem dependências externas. ~8,5k it/s por
  thread (≈60× o Python, que gastava ~45 µs/nó em `observe()`+`info_key`).
  `tests/test_rust_core.py` refaz no Python 6.000 jogos aleatórios gerados pelo
  Rust e exige, a cada decisão, chave igual a `info_key(observe(...))`, mesmas
  ações abstratas legais e mesmo placar final; como a chave Python é invariante
  à informação oculta, a do Rust também é. Regulamentos: `h0` (padrão) e
  `--rules o1` (= `ONLINE_O1`: correr na vez na 1ª rodada de mão normal, mão de
  ferro com cartas visíveis); o teste refaz 6.000 jogos em cada um.
  Diferenças: threads compartilham uma tabela (CFR assíncrono, travas por
  fatia) e o RNG é xoshiro256** (sementes não reproduzem o treino Python);
  com várias threads a ordem de atualização não é determinística.

### Treino turbo (E6): CPU v2 + GPU
- `rust/mccfr/src/v2.rs` (`mccfr train2`): tabela sem travas (open addressing, só hash de
  64 bits), uma consulta por nó, atualizações "Hogwild", filtro de admissão count-min e poda
  por arrependimento estilo Pluribus (`--prune 0.95`).
- `trucoai/gpu/mccfr_kernel.cu` + `cuda.py` + `train.py`: o mesmo motor em CUDA (NVRTC do
  PyTorch, driver via ctypes); travessia em ondas (expand/emit/backup por profundidade).
- `scripts/train_turbo.py`: os dois em paralelo e mescla por visitas. Hash das chaves:
  `fnv1a64-splitmix` (meta do formato compacto); `CompactPolicy` escolhe o hash pela meta.

### Blueprint compacto (`cfr/compact.py`)
Chave `info_key` → blake2b-64 ordenado (`keys.npy`), probabilidades em `uint8`
(`probs.npy`), visitas (`visits.npy`), abertos com `mmap`: ~21 bytes/nó em disco e
páginas compartilhadas entre processos (o pickle ocupava ~480 bytes/nó em RAM, por
processo). Erro de quantização ≤ 1/255. `scripts/compact_blueprint.py` converte.

### Busca com crença (`agents/search.py`, "PIMC+")
- Mundos de `sample_state`; com `belief=1`, sorteia `k·n` e reamostra `n` pela
  verossimilhança (amornada) das APOSTAS que o adversário já fez nesta mão, segundo o
  blueprint (reconstrói o início da mão e reaplica o log público).
- Rollouts guiados pelo blueprint para os dois lados (inclui pedir/aceitar/correr);
  reserva gulosa/equidade em nós pouco visitados; números aleatórios comuns entre ações.
- `mix`: fração das respostas do adversário nos rollouts pelo modelo de equidade
  (robustez contra quem não joga como o blueprint).
- `adaptive`: frequência de corrida do adversário na partida (média bayesiana) ajusta
  a probabilidade de corrida nos rollouts.
- Apostas pelo blueprint em nós com ≥ `bmv` visitas (como o híbrido); cartas pela busca.

### Busca: PIMC
Para cada ação, `n` mundos sorteados → aplica a ação → simula até o fim da mão
com política gulosa de cartas e um modelo de resposta a pedidos baseado em
equidade. Valor = equidade após a mão.

### Híbrido (candidato principal)
Apostas pelo blueprint quando o infoset tem ≥30 visitas; cartas pela busca.

## Próximos passos de pesquisa
1. **Deep CFR / SD-CFR** (EricSteinberger) sobre `Observation` codificada, para
   generalizar entre infosets raros em vez de podar.
2. **MaskablePPO em liga** (sb3-contrib) como baseline neural independente,
   com o blueprint e o heurístico na liga.
3. **Busca com crença**: ponderar `sample_state` pela probabilidade das mãos
   adversárias dado o histórico de apostas (via blueprint) — corrige a principal
   fraqueza do PIMC.
4. **Exploração (best response)** local para medir a explorabilidade do blueprint.
5. **Modelagem de oponente humano** a partir de partidas com consentimento.
