# tesis/ — mediciones reales de prefill y decode (Qwen3-8B, A100-40GB, vLLM 0.20)

Lo que hay aquí es lo que entra en la memoria. Los pilotos y las pruebas de humo siguen en `../resultados/`.

```
tesis/
├── a100_qwen3_8b.sh          sbatch: 1 A100 completa, un solo `vllm serve` y todos los experimentos contra él
├── configs/
│   ├── e1_prefill.toml       E1: TTFT frente a la entrada (128 … 7680), salida 128, 1 usuario, 30 medidas por punto
│   └── e2_concurrencia.toml  E2: TPOT/ITL/throughput frente a la concurrencia (1 … 64), 1024 → 256, 256 medidas por punto
└── resultados/<jobid>/
    ├── entorno.txt  gpu_monitor.csv (1 s)  serve.log  e1_prefill.log  e2_concurrencia.log
    ├── e1-prefill-<fecha>/       serve_in*_out*_c*.json, config.json, RESUMEN.md, metrics_servidor*.txt
    └── e2-concurrencia-<fecha>/  ídem
```

## Lanzar

```bash
cd ~/tfg-jose-ureta
sbatch tesis/a100_qwen3_8b.sh                               # E1 y E2
EXPERIMENTOS=e2_concurrencia sbatch tesis/a100_qwen3_8b.sh  # solo uno
squeue -u $USER
tail -f tesis-8b-<job>.out                                  # fases
tail -f tesis/resultados/<job>/e2_concurrencia.log          # progreso punto a punto
```

La QoS `normal` permite **un trabajo a la vez**: un `salloc` abierto (por ejemplo en el MIG de compute-0-0)
deja este trabajo en `PENDING (QOSMaxJobsPerUserLimit)`. Va a `gpu-dev` (1 GPU, 16 CPU) porque el 27-oct
`sbatch --test-only` en `gpu` estimaba el arranque para el 1-nov y en `gpu-dev` era inmediato.

## Configuración fija de todas las mediciones

| | |
|---|---|
| Modelo | Qwen/Qwen3-8B, BF16, snapshot `b968826d…` de `hf-cache` (`HF_HUB_OFFLINE=1`) |
| Servidor | `vllm serve --max-model-len 8192 --gpu-memory-utilization 0.90 --no-enable-prefix-caching`, con torch.compile y CUDA graphs (sin `--enforce-eager`) |
| Cliente | `tfgbench` en el mismo nodo, contra `127.0.0.1`: el TTFT no incluye red |
| Prompts | ids aleatorios de longitud exacta, semilla 0, sin prefijos compartidos; `ignore_eos`, temperatura 0 |
| Concurrencia | bucle cerrado (como mucho `c` en vuelo); el calentamiento no entra en las métricas |

Definiciones de TTFT, TPOT, ITL y e2el: `../experimentos/README.md` §7.

## Tiempos esperados

- Arranque de `vllm serve`: ~4 min de carga de pesos y **~25 min de torch.compile** (trabajo 30732). Por eso hay
  un solo servidor por trabajo. Si la caché `~/.cache/vllm/torch_compile_cache` sirve para esta configuración, baja mucho.
- E1: ~7 min. E2: ~35 min (c=1 tarda ~15 min: 256 × 3,4 s).
- Límite del trabajo: 3 h.

## Notas para el análisis

- `metrics_servidor*.txt` son contadores **acumulados** desde que arrancó el servidor: los de E2 incluyen E1 y
  el calentamiento. Para obtener el valor de E2, se resta el de E1.
- Referencias teóricas (A100-SXM4-40GB, ~16,4 GB de pesos): decode con lote 1 ≈ 95 tok/s (limitado por el ancho de banda) y
  prefill ≈ 19 000 tok/s (limitado por el cálculo). En el trabajo 30732, `vllm bench latency` dio TPOT ≈ 13,1 ms (≈ 76 tok/s).
