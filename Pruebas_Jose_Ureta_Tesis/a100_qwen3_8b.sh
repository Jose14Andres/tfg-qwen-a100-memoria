#!/bin/bash -l
#SBATCH -J tesis-8b -p gpu-dev -c 16 --mem=96G --gres=gpu:a100-sxm4-40gb:1 -t 03:00:00 -o tesis-8b-%j.out -e tesis-8b-%j.err
# Campaña de la tesis: Qwen3-8B BF16 en una A100-SXM4-40GB completa, vLLM 0.20.
# Un solo `vllm serve` para todo el trabajo (cada arranque cuesta ~25 min de torch.compile) y el cliente tfgbench
# en el mismo nodo (127.0.0.1, sin red de por medio).
#   E1: prefill  (configs/e1_prefill.toml)      E2: concurrencia (configs/e2_concurrencia.toml)
# Resultados en tesis/resultados/<jobid>/. Uso (desde ~/tfg-jose-ureta):  sbatch tesis/a100_qwen3_8b.sh
#   variables: EXPERIMENTOS="e1_prefill e2_concurrencia"  PORT=8000
set -eu
B=$HOME/tfg-jose-ureta
T=$B/tesis
MODEL=Qwen/Qwen3-8B
MAXLEN=8192
PORT=${PORT:-8000}
EXPERIMENTOS=${EXPERIMENTOS:-e1_prefill e2_concurrencia}
R=$T/resultados/${SLURM_JOB_ID:-local}
mkdir -p $R
export HF_HOME=$B/hf-cache HF_HUB_OFFLINE=1
module load python/3.11
cd $R
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# --- Entorno del servidor
source $B/venv-vllm/bin/activate
python -c "import vllm._C" 2>/dev/null || { log "vllm._C no importa: corriendo parche_glibc.sh"; $B/parche_glibc.sh; }
SERVE_ARGS="--host 127.0.0.1 --port $PORT --max-model-len $MAXLEN --gpu-memory-utilization 0.90 --no-enable-prefix-caching"
log "entorno"
{ hostname; date; cat /proc/loadavg
  nvidia-smi -L
  nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm,driver_version --format=csv
  python -c "import vllm,torch;print('vllm',vllm.__version__,'| torch',torch.__version__,'| cuda',torch.version.cuda)"
  echo "MODEL=$MODEL"; echo "vllm serve $MODEL $SERVE_ARGS"; } > entorno.txt 2>&1
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > gpu_monitor.csv &
MON=$!

# --- Servidor
log "vllm serve (arranque con torch.compile: puede tardar ~30 min)"
vllm serve $MODEL $SERVE_ARGS > serve.log 2>&1 &
SRV=$!
trap 'kill $SRV $MON 2>/dev/null || true' EXIT
for i in $(seq 1 540); do
  curl -sf localhost:$PORT/health >/dev/null && break
  kill -0 $SRV 2>/dev/null || { log "vllm serve terminó al arrancar (ver serve.log)"; exit 1; }
  sleep 5
done
curl -sf localhost:$PORT/health >/dev/null || { log "el servidor no arrancó en 45 min (ver serve.log)"; exit 1; }
log "servidor listo"
deactivate

# --- Cliente: un barrido por experimento, todos contra el mismo servidor
source $B/experimentos/.venv/bin/activate
for E in $EXPERIMENTOS; do
  log "$E"
  tfgbench ejecutar --config $T/configs/$E.toml --url http://127.0.0.1:$PORT --directorio $R > $E.log 2>&1 \
    || { log "  $E falló (ver $E.log)"; continue; }
  D=$(ls -d $R/${E//_/-}-* | tail -1)
  cp entorno.txt $D/
  curl -s localhost:$PORT/metrics > $D/metrics_servidor_completo.txt || true
  grep -E "^vllm:(request_prefill_time|request_decode_time|time_to_first_token|inter_token_latency|request_queue_time)_seconds_(sum|count)" \
    $D/metrics_servidor_completo.txt > $D/metrics_servidor.txt || true
  python $B/resumen_pd.py $D
  log "  -> $D/RESUMEN.md"
done
log "FIN -> $R"
