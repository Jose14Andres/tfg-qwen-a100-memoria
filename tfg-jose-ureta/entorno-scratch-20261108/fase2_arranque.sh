#!/bin/bash
# Fase 2: cronometrar el arranque de vllm serve con el entorno copiado a /scratch (activar_scratch.sh).
# Mismos argumentos de servidor que la línea base de /home (docs/linea-base-home-20261102/ensayo.sh): solo cambia el entorno.
# Espera máxima 15 min a /health. Si se excede, NO mata el servidor: informa y sale (decide el usuario).
# Logs en /home (este directorio) porque /scratch/$USER/$SLURM_JOB_ID se borra al terminar el trabajo.
set -u
D=$(dirname "$(readlink -f "$0")")
PORT=${PORT:-8000}
MAX_S=900
cd "$D"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source "$HOME/tfg-jose-ureta/activar_scratch.sh" || { log "FALLO: activar_scratch"; exit 1; }
export HF_HUB_OFFLINE=1
log "entorno: python=$(which python) HF_HOME=$HF_HOME VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-(defecto 600)}"
{ hostname; date; cat /proc/loadavg; nvidia-smi -L; } > entorno.txt 2>&1
curl -s -o /dev/null -w "%{http_code}" localhost:$PORT/health 2>/dev/null | grep -q 200 && { log "FALLO: el puerto $PORT ya responde antes de arrancar"; exit 1; }

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > gpu_monitor.csv 2>&1 &
echo $! > monitor.pid
T0=$(date +%s); log "vllm serve (T0=$(date -d @$T0 +%T))"
nohup vllm serve Qwen/Qwen3-14B --host 127.0.0.1 --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching > serve.log 2>&1 &
SRV=$!; echo $SRV > serve.pid

while :; do
  T=$(( $(date +%s) - T0 ))
  if curl -sf localhost:$PORT/health >/dev/null; then log "/health responde a los ${T} s"; break; fi
  kill -0 $SRV 2>/dev/null || { log "vllm serve terminó al arrancar a los ${T} s (ver serve.log)"; exit 1; }
  [ $T -ge $MAX_S ] && { log "EXCEDIDO: sin /health a los ${T} s; servidor PID $SRV sigue vivo, no lo toco"; exit 2; }
  sleep 1
done
log "nvidia-smi:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
log "/v1/models:"; curl -s localhost:$PORT/v1/models; echo
log "servidor PID $SRV sigue vivo (monitor GPU PID $(cat monitor.pid))"
