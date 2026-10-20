# Integração online reutilizável

O launcher e os módulos de leitura, jogo-sombra, controle da sessão, recibo da observação e cliques estão incluídos nesta revisão. As correções locais v2/v4 foram integradas diretamente ao pacote, sem depender de overlays privados ou de caminhos de outra máquina.

Os nomes são neutros e o endereço é configurado por você. O adaptador continua específico ao protocolo e à interface originalmente observados: fornecer outra URL não o torna compatível com qualquer site. O1 e o fingerprint das regras continuam identificáveis nos recibos históricos.

## Instalar e configurar

```bash
python -m pip install -e ".[browser,dev]"
python scripts/online.py --help
```

É necessário ter Google Chrome instalado; o código usa o canal `chrome` do Playwright. Faça login manualmente na janela aberta. Nenhum perfil ou sessão autenticada é distribuído.

Configure o endereço localmente no PowerShell:

```powershell
$env:TRUCOAI_SITE_URL = "https://seu-endereco-de-mesa/"
python scripts/online.py shadow --agent heuristic
python scripts/online.py play --agent heuristic --max-hands 1
```

A URL acima é um exemplo, não um serviço de jogo. A alternativa é passar `--url` com o endereço real. O endereço não pode conter usuário, senha, query ou fragmento. O programa não lê `.env` automaticamente; use variáveis do processo ou os argumentos.

O padrão `heuristic` existe no pacote e não depende de pesos externos. Para busca, use `--agent pimc:64`. Agentes que exigem blueprints, configurações ou adaptadores não publicados só funcionarão depois que esses artefatos forem restaurados localmente. Isto não disponibiliza o E13 completo por si só nem altera a evidência simulada dos experimentos.

## Modos e sessão

| Modo | Comportamento |
|---|---|
| `shadow` | Acompanha a mesa e registra decisões, sem executar jogadas |
| `play` | Executa decisões, com confirmação pelo protocolo e trava consultada a cada ação |
| `remote` | Recebe decisões manuais no terminal e executa cliques |
| `capture` | Grava dados locais para diagnóstico; imagens apenas com `--screenshots` |

O perfil padrão fica em `.pw-profile/`. Para conectar ao Chrome já aberto com depuração local, use `--cdp http://127.0.0.1:9222`. O navegador precisa ter sido iniciado com depuração e um perfil separado. A aba é selecionada pela origem da URL configurada; as demais abas são preservadas. Fechar o programa não deve encerrar um contexto externo.

A trava existente permanece: sem `--max-stake` nem `TRUCOAI_MAX_STAKE`, somente mesas gratuitas podem receber ações automáticas; valor desconhecido ou desatualizado bloqueia. Não foi restaurado nenhum teto pessoal usado em sessões anteriores. O launcher não escolhe nem inicia uma mesa por conta própria: aguarda o usuário e os eventos reconhecidos.

## Dados locais

`logs/`, `captures/`, `.pw-profile/`, arquivos `.env`, dumps de autenticação e perfis de navegador continuam ignorados pelo Git. Campos de credenciais e identificadores usuais são redigidos nos logs estruturados. `capture` grava payloads e DOM para diagnóstico e pode conter dados privados, inclusive em campos binários ou texto livre: essa gravação é local e não deve ser publicada como se estivesse anonimizada. Screenshots são opcionais e podem mostrar participantes.

## Verificação

Testes sintéticos cobrem URL obrigatória, recusa de credenciais na URL, escolha de aba por origem, redaction de identificadores, transições de partida, memória do agente, mão de onze, confirmação síncrona dos cliques e o recibo exato da observação entregue ao agente. O teste do loop usa navegador substituído por objetos de teste e o agente heuristic realmente carregado.

Essa verificação não abre uma sessão real, não faz login e não joga uma partida. Ela não certifica compatibilidade com a versão atual do serviço. Algumas verificações históricas de políticas e capturas permanecem ignoradas por falta dos artefatos privados.
