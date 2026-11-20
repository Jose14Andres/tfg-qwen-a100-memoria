#!/bin/bash
# Regenera las tablas JSON/Markdown de Ollama, las mete en RESUMEN_TESIS.md y empaqueta todo lo
# necesario para escribir la tesis en ~/tfg-resumen.zip (Jupyter no deja bajar carpetas).
# También regenera las gráficas (graficas/) y las mete en ese zip y, aparte, en ~/tfg-graficas.zip.
# No incluye venv ni pesos de modelos.
set -e
cd "$HOME/tfg-jose-ureta"
python3 ollama/resumir.py > /dev/null
python3 - <<'EOF'
import re
f = "RESUMEN_TESIS.md"
t = open(f).read()
tabla = open("ollama/tabla_ollama.md").read()
t = re.sub(r"<!-- TABLA_OLLAMA -->.*?<!-- /TABLA_OLLAMA -->",
           lambda m: "<!-- TABLA_OLLAMA -->\n" + tabla + "<!-- /TABLA_OLLAMA -->", t, flags=re.S)
open(f, "w").write(t)
EOF
# Gráficas: se regeneran desde los JSON si existe su venv (matplotlib); si no, se empaquetan las que haya.
if [ -x graficas/.venv/bin/python ]; then
  graficas/.venv/bin/python graficas/generar_graficas.py > /dev/null
else
  echo "AVISO: no existe graficas/.venv; se empaquetan las figuras ya generadas sin regenerarlas" >&2
fi
rm -f "$HOME/tfg-resumen.zip" "$HOME/tfg-graficas.zip"
zip -qr "$HOME/tfg-resumen.zip" RESUMEN_TESIS.md \
  ollama/medir_ollama.py ollama/resumir.py ollama/resumen_global.json ollama/tabla_ollama.md ollama/resultados ollama/logs \
  resultados docs graficas -x 'graficas/.venv/*' '*/__pycache__/*'
zip -qr "$HOME/tfg-graficas.zip" graficas -x 'graficas/.venv/*' '*/__pycache__/*'
for z in "$HOME/tfg-resumen.zip" "$HOME/tfg-graficas.zip"; do
  unzip -tq "$z" > /dev/null
  echo "exportado: $z ($(du -h --apparent-size "$z" | cut -f1))"
done
