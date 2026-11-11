#!/bin/bash
# Fase 1: copia del entorno a /scratch del trabajo y verificación del import.
echo "[$(date +%T)] inicio Fase 1 en $(hostname) job $SLURM_JOB_ID"
source ~/tfg-jose-ureta/activar_scratch.sh --preparar || { echo "[$(date +%T)] FALLO preparar"; exit 1; }
echo "[$(date +%T)] verificación:"
which python
python -c "import vllm, sys; print(vllm.__file__); print(sys.prefix)"
echo "HF_HOME=$HF_HOME"
grep -rl "/home/andres.rubio/tfg-jose-ureta/venv-vllm" "$VIRTUAL_ENV/bin" "$VIRTUAL_ENV/pyvenv.cfg" 2>/dev/null | head; echo "(fin de rutas a /home restantes)"
cat "$VIRTUAL_ENV/pyvenv.cfg"
echo "[$(date +%T)] fin Fase 1"
