# Preparação para publicação pública

Preparado em 20/10/2026 como cópia independente. Os diretórios originais main/ e refs/ foram preservados. O repositório começa com histórico novo e não importa objetos Git de clones externos.

## Conteúdo

Incluídos: fontes Python/Rust/CUDA, testes locais, 20 arquivos próprios de pesos .npz, documentação e recibos de simulação E13 com caminhos pessoais removidos.

Excluídos: perfis autenticados, capturas, logs, estatísticas e fixtures de partidas reais, caches, binários, clones, pesos de terceiros, adaptadores com proveniência pendente, blueprints grandes e loader incompleto. Nenhum fonte original foi apagado.

Alterações: apresentação como pesquisa; limitações e proveniência explícitas; descoberta dos subpacotes no empacotamento; erro descritivo para adaptadores ausentes; skips dos testes dependentes de artefatos; uso de random no teste de separação dos grupos; auditoria estatística e CI offline.

## Verificação

Suíte local da cópia pública: 113 testes passaram e 18 foram ignorados por dependências opcionais ou artefatos ausentes. A auditoria E13 confirmou integridade da exportação, contagens e IC pareado, sem reexecutar partidas.

Gitleaks v8.30.1, obtido do release e conferido contra seu checksum, não encontrou credenciais após revisão de cinco falsos positivos. As exceções restritas em .gitleaks.toml são nomes de algoritmos e um SHA256 de tabela NumPy, nunca credenciais.

A inspeção dos arquivos selecionados e metadados textuais dos 20 .npz não apontou caminhos pessoais ou credenciais. research/publication_manifest.json registra os arquivos selecionados, exceto o próprio manifesto. O commit usa bytes preservados para manter os hashes dos recibos. Uma exportação do commit é validada separadamente dos caches locais antes da publicação.

## Limites

Varredura automática não garante ausência absoluta de dados sensíveis. Não há licença permissiva global aplicada, DOI, revisão por pares, reprodução integral do E13 ou demonstração de desempenho humano. Os problemas de proveniência identificados foram tratados por exclusão.

## Integração contínua

O token disponível permite criar e publicar o repositório, mas o GitHub recusou a criação de workflow por ausência do escopo workflow. O modelo está em docs/ci/research.yml; nenhum CI foi ativado nesta publicação. Os comandos foram validados localmente sobre git archive. Para ativar futuramente, instale esse modelo em .github/workflows/research.yml com uma autenticação autorizada.

## Revisão da integração online

Adicionados launcher e módulos de integração com nomes neutros e endereço fornecido localmente. As correções locais v2/v4 foram integradas no pacote. Perfis, credenciais, capturas, logs pessoais e tetos de sessões anteriores não foram incluídos. A árvore atual não contém o nome ou domínio da plataforma. Commits e releases anteriores preservam o histórico público e não foram reescritos. A referência das regras permanece O1; a neutralização nominal não implica compatibilidade com qualquer plataforma.

Verificação da revisão 0.2.0: 144 testes passaram, 22 foram ignorados; quatro testes novos do launcher falharam antes da implementação e passaram após ela. Gitleaks não encontrou credenciais na árvore em preparação. A auditoria estatística E13 continua conferindo os mesmos resultados; nomes neutralizados são transformações da exportação, não novos jogos.
