#!/bin/bash
# Ensayo de punta a punta: mismos argumentos de servidor que tesis/a100_qwen3_14b.sh, carga mínima, sin tocar tesis/resultados.
set -u
B=$HOME/tfg-jose-ureta
D=$(dirname "$(readlink -f "$0")")
PORT=8000
cd "$D"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
module load python/3.11
export HF_HOME=$B/hf-cache HF_HUB_OFFLINE=1
source $B/venv-vllm/bin/activate
log "vllm serve"
vllm serve Qwen/Qwen3-14B --host 127.0.0.1 --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.90 --no-enable-prefix-caching > serve.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 1080); do
  curl -sf localhost:$PORT/health >/dev/null && break
  kill -0 $SRV 2>/dev/null || { log "vllm serve terminó al arrancar"; exit 1; }
  sleep 5
done
curl -sf localhost:$PORT/health >/dev/null || { log "no arrancó en 90 min"; exit 1; }
log "servidor listo"
grep -iE "KV cache|Maximum concurrency" serve.log | tail -2
curl -s localhost:$PORT/v1/models | head -c 300; echo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
deactivate
source $B/experimentos/.venv/bin/activate
log "tfgbench"
tfgbench ejecutar --config $D/ensayo_14b.toml --url http://127.0.0.1:$PORT --directorio $D/res > tfgbench.log 2>&1; log "tfgbench exit=$?"
R=$(ls -d $D/res/ensayo-14b-* | tail -1); log "dir=$R"; ls -la "$R"
curl -s localhost:$PORT/metrics > $R/metrics_servidor_completo.txt
grep -E "^vllm:(request_prefill_time|request_decode_time|time_to_first_token|inter_token_latency|request_queue_time)_seconds_(sum|count)" $R/metrics_servidor_completo.txt > $R/metrics_servidor.txt
python $B/resumen_pd.py $R; log "resumen exit=$?"
cat $R/RESUMEN.md
log "FIN"
