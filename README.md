# TrucoAI â€” pesquisa em Truco Paulista 1v1

Pesquisa experimental sobre agentes de informaÃ§Ã£o imperfeita para Truco Paulista com dois jogadores individuais. Inclui motor de regras, observaÃ§Ãµes por jogador, MCCFR, busca, polÃ­ticas neurais e avaliaÃ§Ã£o contra bots. A PlayTruco Ã© a referÃªncia de modalidade; este projeto Ã© independente, sem afiliaÃ§Ã£o declarada com a plataforma.

**Status: pesquisa em andamento, sem revisÃ£o por pares.** Os resultados publicados sÃ£o de simulaÃ§Ã£o. NÃ£o demonstram forÃ§a contra humanos, equilÃ­brio de Nash, optimalidade ou fidelidade integral ao site.

## Resultado selecionado: E13

No painel fixo de sete adversÃ¡rios, E13 venceu **363/560 partidas (64,82%)**, contra **319/560 (56,96%)** do controle E9. Ganho pareado: **7,86 pontos percentuais**, IC normal de 95% **[2,02; 13,70]**. Houve 1.120 partidas principais e 160 exploratÃ³rias, com posiÃ§Ãµes espelhadas e sementes comuns aos candidatos.

E13 fez **47,5% contra Zap v1**: a mÃ©dia do painel nÃ£o implica vencer cada adversÃ¡rio. Tempo mÃ©dio registrado: 0,302 s por decisÃ£o; mÃ¡ximo observado: 0,938 s. As latÃªncias dependem da mÃ¡quina e do protocolo. [MÃ©todo, tabela completa e limites](docs/RESEARCH.md).

## Executar localmente

Python 3.10 ou superior; use um ambiente virtual.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev,learned]"
python -m pytest -q
python scripts/audit_e13.py
python scripts/evaluate.py --agents random heuristic --deals 20 --workers 1 --tag local
python scripts/play_cli.py --agent heuristic
```

Os dois Ãºltimos comandos usam o padrÃ£o histÃ³rico `h0`; nÃ£o reproduzem o E13/O1. Para criar um blueprint novo sob O1:

```bash
python scripts/train_mccfr.py --help
python scripts/train_mccfr_rs.py --help
```

Treinar redes exige o extra `training`; CUDA e compilador Rust sÃ£o opcionais. Os testes dependentes de blueprints histÃ³ricos ficam explicitamente ignorados quando esses artefatos estÃ£o ausentes. Veja [reproduÃ§Ã£o](docs/REPRODUCIBILITY.md).

## ConteÃºdo e fronteiras

- `trucoai/`: regras, observaÃ§Ã£o legÃ­tima, agentes, treino e componentes de integraÃ§Ã£o.
- `tests/`: testes locais; nÃ£o incluem capturas de participantes.
- `rust/mccfr/`: fonte do nÃºcleo de treino; sem binÃ¡rios prÃ©-compilados.
- `runs/*/policy.npz`: pesos produzidos pelos experimentos locais; nÃ£o incluem checkpoints de terceiros.
- `research/e13/`: exportaÃ§Ã£o dos recibos de simulaÃ§Ã£o, com caminhos locais removidos e manifesto de transformaÃ§Ã£o.

```mermaid
flowchart LR
    accTitle: Fronteira de informaÃ§Ã£o do agente
    accDescr: O ambiente fornece apenas a observaÃ§Ã£o legÃ­tima ao agente e valida a aÃ§Ã£o retornada.
    E[Estado completo no ambiente] --> O[ObservaÃ§Ã£o legÃ­tima]
    O --> A[Agente]
    A --> V[ValidaÃ§Ã£o de aÃ§Ã£o legal]
    V --> E
```

O regulamento O1 e suas hipÃ³teses estÃ£o em [REGRAS.md](docs/REGRAS.md). Relatos histÃ³ricos de observaÃ§Ã£o do site nÃ£o sÃ£o uma verificaÃ§Ã£o atual. Esta distribuiÃ§Ã£o nÃ£o inclui perfil de navegador, sessÃµes autenticadas, capturas reais, dados de participantes nem launcher de jogo no site. A integraÃ§Ã£o completa com humanos permanece fora da validaÃ§Ã£o desta publicaÃ§Ã£o.

[ProveniÃªncia e direitos](docs/PROVENANCE.md) Â· [Privacidade](docs/PRIVACY.md) Â· [PreparaÃ§Ã£o para publicaÃ§Ã£o](docs/PUBLICATION.md)
