#!/bin/bash
# Regenera las tablas JSON/Markdown de Ollama, las mete en RESUMEN_TESIS.md y empaqueta todo lo
# necesario para escribir la tesis en ~/tfg-resumen.zip (Jupyter no deja bajar carpetas).
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
rm -f "$HOME/tfg-resumen.zip"
zip -qr "$HOME/tfg-resumen.zip" RESUMEN_TESIS.md \
  ollama/medir_ollama.py ollama/resumir.py ollama/resumen_global.json ollama/tabla_ollama.md ollama/resultados ollama/logs \
  resultados docs
echo "exportado: $HOME/tfg-resumen.zip ($(du -h "$HOME/tfg-resumen.zip" | cut -f1))"
