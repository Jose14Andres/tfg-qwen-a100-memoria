# Resumen acumulado del TFG — medición de prefill y decode en una A100

Se actualiza tras cada tanda de pruebas. Solo números medidos; lo que no está diagnosticado se dice.
Datos crudos en JSON junto a cada resultado (para las gráficas). Metodología: prueba y error.

## Entorno

| elemento | valor |
|---|---|
| Clúster | HPC CEDIA, Slurm; nodos compute-0-1 / compute-0-2 (A100-SXM4-40GB completa), driver 535 |
| Particiones | `interactive` suele dar trozos MIG 1g.5gb (5 GB); `gpu`/`gpu-dev` la A100 entera |
| vLLM | 0.20.0+cu129, torch 2.11, en `venv-vllm` (necesita `parche_glibc.sh`: la rueda pide GLIBC 2.34, el nodo tiene 2.31) |
| Ollama | 0.24.0 + capa cuda-compat 12.8 en `~/ollama` (≥0.30 exige driver ≥550) |

## Cronología

| fecha | qué | resultado medido |
|---|---|---|
| 21–24 oct | instalación de vLLM, prueba de humo | funciona tras el parche glibc; Qwen3-0.6B corre incluso en MIG 5 GB |
| 24 oct | barrido prefill/decode `bench_pd.sh` (job 30732) | TIMEOUT a las 5 h; la fase in=128 sola tardó ~2,5 h; solo 3 puntos (sin resumen) |
| 25–27 oct | cliente propio `tfgbench` (streaming, JSON compatible con `vllm bench serve`) | piloto completo con Qwen3-0.6B en MIG 5 GB → `resultados/piloto-mig-0.6b-20261027-155013/` |
| 27 oct | campaña Qwen3-8B (job 31157) | falló: el entorno tardó 21 min en prepararse y `vllm serve` no escribió nada en 45 min |
| 29 oct | el tutor pide un modelo mayor → Qwen3-14B BF16 (28 GB) | descargado; shell colgada por `spack load hwloc` en `.bashrc` (corregido) |
| 2 nov | línea base: arranque de `vllm serve` 14B desde /home | murió a los 59 min 31 s por el límite interno de vLLM (`VLLM_ENGINE_READY_TIMEOUT_S` = 600 s), 4 min 19 s después de cargar los pesos |
| 8 nov | Fase 1: copiar el entorno a /scratch local (job 32710) | hf-cache copiada en 459 s; el log se corta antes de terminar la copia del venv |
| 11 nov | Fase 1 repetida (job 33107) | hf-cache copiada en 414 s; del venv (12 GB) solo 1,2 GB copiados a las 18:02 (71 min después de empezar), con loadavg ~790 |
| 11 nov | vLLM 14B en pausa por decisión del usuario; se pasa a medir Ollama | ver abajo |

## Ollama (11-nov-2026, job 33107, compute-0-1, A100 40 GB completa)

**Método** (`ollama/medir_ollama.py`, solo stdlib): API nativa `/api/generate` en streaming, `raw`, temperatura 0,
`num_ctx` 8192, una solicitud a la vez (`OLLAMA_NUM_PARALLEL=1`), 1 calentamiento + 5 repeticiones por punto,
128 tokens de salida. Cada prompt lleva un prefijo único para que no se reutilice la caché KV.
TTFT se mide en el cliente; prefill y TPOT son los que reporta Ollama (`prompt_eval_duration`, `eval_duration`).
Valores = medianas.

**Tabla** (se regenera con `python3 ollama/resumir.py` → `ollama/tabla_ollama.md` y `ollama/resumen_global.json`):

<!-- TABLA_OLLAMA -->
| modelo | prompt_tokens | ttft_ms | prefill_ms | prefill_tok_s | tpot_ms | decode_tok_s | latencia_s | n | fallos |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-coder:30b | 195 | 1758.0 | 100.1 | 1947.4 | 8.5 | 117.5 | 2.9 | 5 | 0 |
| qwen3-coder:30b | 717 | 1608.0 | 270.2 | 2653.7 | 8.7 | 115.1 | 2.8 | 5 | 0 |
| qwen3-coder:30b | 2800 | 2824.0 | 918.0 | 3050.2 | 9.1 | 109.7 | 4.0 | 5 | 0 |
<!-- /TABLA_OLLAMA -->

**Observaciones medidas**
- qwen3-coder:30b: carga inicial 85,8 s (pesos leídos desde /home por NFS); ocupa 18,9 GB de GPU.
- El TTFT del cliente supera al prefill del servidor en 1,0–2,4 s por solicitud. Esa diferencia coincide
  (±10 ms) con el `load_duration` que reporta Ollama en cada solicitud, aunque el modelo ya estaba cargado
  (memoria GPU constante). La red/cliente aporta 1 ms. **Causa sin diagnosticar**; el nodo tenía loadavg 260–380.
- El TPOT es muy estable (desviación ≤ 0,07 ms) y crece poco con el contexto: 8,5 → 9,1 ms de 195 a 2800 tokens.
- El prefill escala con el prompt: 1947 → 3050 tok/s de 195 a 2800 tokens.
- nemotron-3-nano:30b (Q4_K_M, 24,3 GB), lanzado justo después: carga de 226,8 s (runner listo en 172,7 s).
  Ollama no pudo refrescar la memoria libre de la GPU (`unable to refresh free memory, using old values`) y,
  con el dato viejo (20,4 GiB libres, con el qwen3-coder aún contado), metió solo **43 de 53 capas en GPU**
  (17,9 GiB) y **4,7 GiB en CPU**. Tras cargar, la primera solicitud de calentamiento no había terminado
  en 54 min (17:08 → 18:02), con la GPU al 0 % y loadavg ~790. **Medición no válida** (descarga parcial a CPU).
  Lección: entre modelos, reiniciar Ollama para que lea la memoria libre real.

## Pendiente
- Diagnosticar el `load_duration` por solicitud de Ollama.
- vLLM con Qwen3-14B: Fase 2 (cronometrar el arranque desde /scratch) cuando el usuario lo decida.
