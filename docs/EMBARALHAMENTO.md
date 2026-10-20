# Embaralhamento — referência PlayTruco, verificada em 20/10/2026

Fontes públicas lidas no navegador:

- https://www.playtruco.com/embaralhamento — página «Embaralhamento justo»,
  incluindo as etapas RNG certificado, embaralhamento e distribuição.
- https://access.gaminglabs.com/certificate/index?i=610 — certificado de
  Marketing Play Business LTDA dba playtruco.

A página descreve o princípio Fisher–Yates aplicado ao baralho inteiro com
aleatoriedade de um RNG auditado; a distribuição ocorre depois. Não publica
o código do gerador, suas sementes nem a ordem exata dos índices de entrega.
Não presumimos padrões previsíveis ou informação antecipada disponível à IA.

O certificado identifica «MPB Random Number Generator (RNG)», resultado PASS,
data de certificação 28/10/2025 e validade até o último dia de novembro de 2026.
Seu escopo é a avaliação do produto RNG listado; não certifica nosso simulador
nem publica o algoritmo interno. A própria página limita o significado do
certificado e não o equipara a endosso ou garantia de desempenho.

## Contrato no experimento E8

`GameState.deal` embaralha as 40 cartas com `random.Random.shuffle` antes de
distribuir. Inspeção do Python 3.14.2 instalado confirmou Fisher–Yates
descendente, índices uniformes `0..i`, gerados por bits com rejeição de valores
fora do intervalo. Assim evitamos duplicatas e viés por redução modular.

O PRNG local é Mersenne Twister com sementes registradas para reproduzir o
experimento. Ele **não é o RNG certificado da PlayTruco**, não é criptográfico
e não tenta reproduzir as cartas reais do site. O que reproduzimos é o
contrato probabilístico de permutação uniforme sem reposição. Sementes,
baralho futuro e cartas adversárias nunca entram nas features da política.

O sorteio de mundos na busca usa somente cartas ainda possíveis após excluir
as conhecidas. Ponderar esses mundos por ações públicas do adversário é
inferência condicional; não altera o embaralhamento do ambiente nem acessa
as cartas ocultas verdadeiras.

A ordem alternada de entrega e a vira após seis cartas continuam sendo a
convenção já existente no motor: a página de embaralhamento não confirma
esses índices. Para uma permutação uniforme, qualquer atribuição fixa a sete
posições distintas mantém a mesma distribuição conjunta das cartas.

Verificação: `tests/test_shuffle.py` rastreia as 39 trocas, confere a entrega
após embaralhar, a unicidade e reprodução por semente, e enumera todas as 24
permutações de um exemplo de quatro cartas. Isso testa o contrato do
simulador; não é certificação estatística ou auditoria do servidor.

Os dados E8 já em coleta usam esse mesmo caminho Python e permanecem válidos
sob esse contrato. Checkpoints históricos Rust/GPU ficam preservados; esta
verificação não atribui identidade de PRNG entre essas implementações.
