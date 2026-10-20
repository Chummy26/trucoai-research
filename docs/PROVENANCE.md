# Proveniência e direitos

A publicação contém uma seleção do diretório local de pesquisa, não os clones usados como referência. Não foi aplicada licença permissiva global: a titularidade e as permissões precisam ser resolvidas antes de conceder direitos de reutilização. Repositório público não significa autorização automática para redistribuir ou relicenciar conteúdo.

## Referências históricas

| Projeto | Revisão local registrada | Uso relatado |
|---|---|---|
| [truco-de-malandro](https://github.com/lgfauth/truco-de-malandro) | 6c4102c | Comparação do motor, RulePlayer e adaptador PPO |
| [truco-mccfr-ai](https://github.com/filevich/truco-mccfr-ai) | a464bf2 | Heurísticas Simple/SimpleX adaptadas |
| [Deep-CFR](https://github.com/EricSteinberger/Deep-CFR) | 2e664eb | Referência de métodos CFR |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | 4840189 | Referência de ambiente e informação imperfeita |
| [RLCard](https://github.com/datamllab/rlcard) | d7d0a95 | Referência de interface de agentes |
| [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | f094181 | Referência de política com máscara |
| [Truco-AI](https://github.com/RenzoDavila27/Truco-AI) | 6de100b | Referência de outra variante; não é baseline equivalente |

Os documentos locais eram inconsistentes: descreviam ausência de cópia literal e, adiante, um port literal de RulePlayer. Os clones locais de truco-de-malandro e truco-mccfr-ai não apresentavam arquivo LICENSE na raiz consultada. Por isso `agents/references/`, seus checkpoints e o teste de equivalência com os originais foram excluídos da publicação. Isso é uma decisão conservadora de distribuição, não uma conclusão jurídica sobre os projetos.

As referências são créditos históricos, não uma alegação de revisão atual das páginas. Nenhuma biblioteca, clone, recurso visual ou código cliente da PlayTruco é redistribuído. Componentes de protocolo são código local de integração; não demonstram afiliação ou permissão de operação no serviço. Pesos em `runs/` foram gerados pelo treino local conforme os recibos; não incluem `ref_weights` ou modelos PPO obtidos de terceiros.

Os demais fontes preservam suas referências de método. Esta preparação não certifica originalidade linha por linha nem resolve a titularidade de todo o código. Solicitações de correção de crédito devem identificar arquivo e origem em uma issue pública sem dados privados.
