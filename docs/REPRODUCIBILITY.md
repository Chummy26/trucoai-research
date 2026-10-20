# Reprodução

## Disponível

Motor Python, observação, MCCFR, buscas, modelos e testes sem artefatos externos. `python -m pytest -q` verifica a cópia pública; os skips por blueprint ausente são explícitos em `tests/conftest.py`. CUDA, executável Rust, checkpoints ou dependências opcionais ausentes também podem provocar skips próprios. O teste de separação de grupos usa o agente local random em lugar de um adaptador omitido; sua finalidade continua sendo verificar ausência de sobreposição de partidas.

`python scripts/audit_e13.py` confere os hashes do manifesto público e recalcula as vitórias e o intervalo do ganho a partir dos 40 blocos pareados por adversário. Preserva-se a estrutura dos recibos; o prefixo do diretório pessoal foi removido. `original_sha256` identifica os bytes privados de origem e `public_sha256` identifica o arquivo exportado. Os hashes embutidos dos recibos históricos não foram recalculados para fingir que a exportação é o original.

## Indisponível

A arena histórica E13 exige blueprint MCCFR, configurações congeladas e adaptadores omitidos. O pacote não fornece uma política E13 pronta para inferência, não usa fallback silencioso e não publica o loader incompleto. Arquivos `.npz` próprios estão preservados, mas não bastam para reconstruir a política composta histórica.

Os defaults históricos de vários scripts usam h0. E13 usou O1, fingerprint `ca73a2770a14c415`. Não compare um treino h0 como se fosse repetição de O1. Um novo treino tem outra identidade e não deve sobrescrever resultados antigos.

## Ambiente de verificação da publicação

Windows, Python 3.14; pytest 9.0.2, NumPy 2.4.2, PyTorch 2.10.0+cu126. As dependências mínimas do pacote usam intervalos; estas são versões observadas, não garantia de bit equivalência em qualquer plataforma. Use `requirements-research.txt` para as versões de pytest/NumPy verificadas. PyTorch e CUDA são opcionais, separados desse arquivo.

## Integração reutilizável

O launcher scripts/online.py e as correções de integração foram acrescentados com endereço configurável e nomes neutros. Veja ONLINE_INTEGRATION.md. Isso permite reconstruir o fluxo com agentes disponíveis; as limitações de reprodução da política composta E13 permanecem. Os recibos exportados também neutralizam nomes e caminhos de módulos; os hashes originais embutidos não identificam os novos fontes.
