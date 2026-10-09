#!/usr/bin/env bash
# Baixa a fonte externa de validação (ADR-018):
# Kaggle mexwell/long-distance-running-dataset — CC BY 4.0.
#
# O endpoint público dispensa credenciais do Kaggle. São 394 MB compactados e
# 2,3 GB descompactados; os arquivos NÃO são versionados.
#
# Destino: $DIRETORIO_DADOS_EXTERNOS ou ~/.cache/tcc-back-2/dados
set -euo pipefail

URL="https://www.kaggle.com/api/v1/datasets/download/mexwell/long-distance-running-dataset"
DESTINO="${DIRETORIO_DADOS_EXTERNOS:-$HOME/.cache/tcc-back-2/dados}"
ZIP="$DESTINO/long-distance-running.zip"

# Só os arquivos de corrida; os CSVs de COVID não são usados pelos experimentos.
ARQUIVOS=(
  run_ww_2019_d.csv run_ww_2019_w.csv
  run_ww_2020_d.csv run_ww_2020_w.csv
)

mkdir -p "$DESTINO"

faltando=0
for arquivo in "${ARQUIVOS[@]}"; do
  [[ -f "$DESTINO/$arquivo" ]] || faltando=1
done

if [[ $faltando -eq 0 ]]; then
  echo "Todos os arquivos já estão em $DESTINO — nada a fazer."
  exit 0
fi

echo "Baixando ~394 MB para $DESTINO (pode levar alguns minutos)..."
curl -fL --progress-bar -o "$ZIP" "$URL"

echo "Extraindo os arquivos de corrida..."
unzip -o -q "$ZIP" "${ARQUIVOS[@]}" -d "$DESTINO"
rm -f "$ZIP"

echo
echo "Pronto. Arquivos em $DESTINO:"
du -h "$DESTINO"/run_ww_*.csv
