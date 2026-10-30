#!/bin/bash -l
#SBATCH --job-name=limite
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=00:40:00
#SBATCH --output=limite-%j.out
cd ~/tfg-jose-ureta
module load python/3.11
source venv-vllm/bin/activate
nvidia-smi -L
vllm serve Qwen/Qwen3-8B --max-model-len 8192 --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 --enforce-eager --port 18000 &
for i in $(seq 1 90); do
  curl -sf http://127.0.0.1:18000/health >/dev/null && break || sleep 10
done
MODEL=Qwen/Qwen3-8B python ttft.py 4096 128
pkill -f "vllm serve"
