# Blueprint MCCFR O1

O blueprint histórico está disponível como asset da [release v0.2.0](https://github.com/Chummy26/trucoai-research/releases/tag/v0.2.0), separado do histórico Git. O pacote contém somente quatro arquivos da tabela: keys.npy, probs.npy, visits.npy e meta.json, sob compact/. Nenhum perfil, sessão, identificador de participante ou captura é incluído.

Tamanho comprimido: 88.722.464 bytes; original: 204.618.326 bytes. Compressão sem perda, com SHA-256 conferido para os quatro arquivos. O manifesto versionado é [blueprint_o1_s7.json](../research/blueprint_o1_s7.json).

## Baixar e restaurar

Na raiz do clone, usando PowerShell e GitHub CLI:

```powershell
gh release download v0.2.0 --repo Chummy26/trucoai-research --pattern "blueprint_o1_s7.*" --dir artifacts
$expected = "b47aac2c56631db2211d3fe305ddd5d64f06e8db55fae51d2d81045fad77634e"
$actual = (Get-FileHash artifacts/blueprint_o1_s7.tar.xz -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) { throw "Checksum do blueprint incorreto" }
python -m tarfile -e artifacts/blueprint_o1_s7.tar.xz runs/mccfr_rs_o1_s7
```

Extraia em uma pasta nova se já houver um compact/ com trabalho local. O comando acima restaura runs/mccfr_rs_o1_s7/compact/. O arquivo é público e também pode ser baixado pela interface de Releases.

Com as dependências instaladas, o alias `zap` consegue carregar essa tabela. Por exemplo:

```powershell
python scripts/online.py shadow --agent zap --url "https://seu-endereco-de-mesa/"
```

O endereço é apenas um exemplo e deve ser substituído localmente. Esse comando usa Zap v1, não o E13. O E13 composto ainda exige configurações e adaptadores não distribuídos; publicar esta tabela não restaura esses componentes. Os resultados históricos continuam sendo de simulação.
