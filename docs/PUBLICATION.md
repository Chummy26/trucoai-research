# PreparaÃ§Ã£o para publicaÃ§Ã£o pÃºblica

Preparado em 20/10/2026 como cÃ³pia independente. Os diretÃ³rios originais main/ e refs/ foram preservados. O repositÃ³rio comeÃ§a com histÃ³rico novo e nÃ£o importa objetos Git de clones externos.

## ConteÃºdo

IncluÃ­dos: fontes Python/Rust/CUDA, testes locais, 20 arquivos prÃ³prios de pesos .npz, documentaÃ§Ã£o e recibos de simulaÃ§Ã£o E13 com caminhos pessoais removidos.

ExcluÃ­dos: perfis autenticados, capturas, logs, estatÃ­sticas e fixtures de partidas reais, caches, binÃ¡rios, clones, pesos de terceiros, adaptadores com proveniÃªncia pendente, blueprints grandes e loader incompleto. Nenhum fonte original foi apagado.

AlteraÃ§Ãµes: apresentaÃ§Ã£o como pesquisa; limitaÃ§Ãµes e proveniÃªncia explÃ­citas; descoberta dos subpacotes no empacotamento; erro descritivo para adaptadores ausentes; skips dos testes dependentes de artefatos; uso de random no teste de separaÃ§Ã£o dos grupos; auditoria estatÃ­stica e CI offline.

## VerificaÃ§Ã£o

SuÃ­te local da cÃ³pia pÃºblica: 113 testes passaram e 18 foram ignorados por dependÃªncias opcionais ou artefatos ausentes. A auditoria E13 confirmou integridade da exportaÃ§Ã£o, contagens e IC pareado, sem reexecutar partidas.

Gitleaks v8.30.1, obtido do release e conferido contra seu checksum, nÃ£o encontrou credenciais apÃ³s revisÃ£o de cinco falsos positivos. As exceÃ§Ãµes restritas em .gitleaks.toml sÃ£o nomes de algoritmos e um SHA256 de tabela NumPy, nunca credenciais.

A inspeÃ§Ã£o dos arquivos selecionados e metadados textuais dos 20 .npz nÃ£o apontou caminhos pessoais ou credenciais. research/publication_manifest.json registra os arquivos selecionados, exceto o prÃ³prio manifesto. O commit usa bytes preservados para manter os hashes dos recibos. Uma exportaÃ§Ã£o do commit Ã© validada separadamente dos caches locais antes da publicaÃ§Ã£o.

## Limites

Varredura automÃ¡tica nÃ£o garante ausÃªncia absoluta de dados sensÃ­veis. NÃ£o hÃ¡ licenÃ§a permissiva global aplicada, DOI, revisÃ£o por pares, reproduÃ§Ã£o integral do E13 ou demonstraÃ§Ã£o de desempenho humano. Os problemas de proveniÃªncia identificados foram tratados por exclusÃ£o.

## Integração contínua

O token disponível permite criar e publicar o repositório, mas o GitHub recusou a criação de workflow por ausência do escopo workflow. O modelo está em docs/ci/research.yml; nenhum CI foi ativado nesta publicação. Os comandos foram validados localmente sobre git archive. Para ativar futuramente, instale esse modelo em .github/workflows/research.yml com uma autenticação autorizada.
