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
| 16 nov | Ollama nemotron-3-nano:30b repetido con el servidor recién arrancado, dentro del contenedor enroot `jos_env` (job 36268) | **medición válida**: 53/53 capas en GPU, 0 fallos; TPOT 6,5–6,6 ms (≈152 tok/s); carga del modelo 181,9 s (ver abajo) |
| 17 nov | Ollama qwen3-coder:30b, intento de repetición (job 36509) | **fallida**: Ollama no detectó la GPU (descubrimiento agotado a los 30 s) y cargó el modelo con 0/49 capas en GPU, todo en CPU; no se lanzó ninguna medición (ver abajo) |
| 19 nov | vLLM 14B desde /home con `VLLM_ENGINE_READY_TIMEOUT_S=3600` (job 37415) | sin resultado: 25 min 48 s hasta el primer mensaje de vLLM; la reserva terminó a los 32 min 13 s, antes de que arrancara EngineCore; GPU a 0 MiB todo el rato (ver abajo) |

## Ollama (11-nov-2026 job 33107 y 16-nov-2026 job 36268, compute-0-1, A100 40 GB completa)

**Método** (`ollama/medir_ollama.py`, solo stdlib): API nativa `/api/generate` en streaming, `raw`, temperatura 0,
`num_ctx` 8192, una solicitud a la vez (`OLLAMA_NUM_PARALLEL=1`), 1 calentamiento + 5 repeticiones por punto,
128 tokens de salida. Cada prompt lleva un prefijo único para que no se reutilice la caché KV.
TTFT se mide en el cliente; prefill y TPOT son los que reporta Ollama (`prompt_eval_duration`, `eval_duration`).
Valores = medianas.

**Tabla** (se regenera con `python3 ollama/resumir.py` → `ollama/tabla_ollama.md` y `ollama/resumen_global.json`):

<!-- TABLA_OLLAMA -->
| modelo | prompt_tokens | ttft_ms | prefill_ms | prefill_tok_s | tpot_ms | decode_tok_s | latencia_s | n | fallos |
|---|---|---|---|---|---|---|---|---|---|
| nemotron-3-nano:30b | 170 | 983.1 | 112.3 | 1513.6 | 6.6 | 152.0 | 1.9 | 5 | 0 |
| nemotron-3-nano:30b | 615 | 1599.1 | 278.9 | 2204.8 | 6.5 | 153.4 | 2.5 | 5 | 0 |
| nemotron-3-nano:30b | 2390 | 2554.1 | 794.8 | 3007.2 | 6.6 | 152.1 | 3.4 | 5 | 0 |
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

**nemotron-3-nano:30b, repetición válida (16-nov-2026, job 36268)**

*Qué cambió respecto al 11-nov:* Ollama se arrancó de cero solo para este modelo (GPU a 0 MiB antes de cargar) y, a
petición del usuario, dentro del contenedor enroot `jos_env` (Ubuntu 20.04.6, glibc 2.31, cuda-compat 12.8). El
cliente `medir_ollama.py` corre en el host del nodo (el contenedor no tiene python3) contra 127.0.0.1:11434. Mismo
método y parámetros que el 11-nov. Detalle en `ollama/resultados/nemotron-3-nano_30b-20261116-185849/entorno.json`;
logs en `ollama/logs/ollama_serve_36268.log` y `ollama/logs/medir_nemotron-3-nano_36268.log`.

*Medido:*
- Carga: **53 de 53 capas en GPU** (pesos 22,4 GiB en GPU + 231 MiB en CPU, caché KV 1,2 GiB, total 24,0 GiB según
  el log). Runner listo en 174,9 s; primera respuesta del cliente a los 181,9 s. GPU usada después: 24 819 MiB.
  El aviso `unable to refresh free memory, using old values` volvió a salir, pero con el servidor recién arrancado
  el valor viejo era el real (GPU vacía), así que esta vez no hubo descarga a CPU.
- 18 solicitudes (3 longitudes × 1 calentamiento + 5 repeticiones), **0 fallos**, 128 tokens de salida en todas.
- loadavg del nodo (1 min): 85,7 al empezar y 88,9 al terminar (frente a 333 el 11-nov).
- Duración de la medición: 18:58:49 → 19:02:42 (3 min 53 s, de ellos 3 min 2 s de carga).
- **TPOT 6,5–6,6 ms (≈152 tok/s)**, prácticamente constante de 170 a 2390 tokens de prompt (6,52–6,58 ms de
  mediana). Es un 23–28 % menor que el de qwen3-coder:30b (8,5–9,1 ms).
- Prefill: 1514 → 2205 → 3007 tok/s para 170 / 615 / 2390 tokens, con dispersión mínima (p. ej. 793–796 ms a 2390).
- **El `load_duration` por solicitud también aparece aquí:** 0,75–2,20 s por solicitud con el modelo ya cargado
  (mismo comportamiento que qwen3-coder el 11-nov). TTFT − prefill − load_duration = 2,3–7,0 ms. O sea, se repite
  con un loadavg ~4 veces menor y dentro del contenedor; la causa sigue **sin diagnosticar**.

*Comparabilidad:* los dos modelos se tokenizan distinto (las mismas 128/512/2048 palabras dan 195/717/2800 tokens en
qwen3-coder y 170/615/2390 en nemotron), por eso las gráficas usan tokens en el eje X. Además, qwen3-coder (11-nov)
se midió antes de pasar al contenedor (su registro no anota el entorno) y con el nodo mucho más cargado.

**qwen3-coder:30b, intento de repetición fallido (17-nov-2026, job 36509)**

Único registro: `ollama/logs/ollama_serve_36509.log` (no hay log del cliente ni carpeta de resultados).
- Reserva: `salloc` en la partición `gpu`, compute-0-1, 07:48:49 → 08:27:32, estado `COMPLETED` (38 min 43 s de 2 h).
- `ollama serve` (0.24.0) arrancó a las 08:09:06. El primer intento de descubrir GPUs falló a las 08:09:36 con
  `failure during GPU discovery ... failed to finish discovery before timeout`; el segundo solo encontró la CPU
  (`total_vram="0 B"`, contexto por defecto 4096 en vez de 8192).
- A las 08:15:21 se pidió cargar qwen3-coder:30b (Q4_K_M): **0 de 49 capas en GPU**, 17,7 GiB en CPU, 128 hilos.
  El runner tardó 180,2 s en estar listo (08:18:22).
- El log no tiene ninguna solicitud `POST` (ni carga desde el cliente ni mediciones); solo un `GET /api/ps` a las
  08:22:08. **Sin resultados.**
- **Sin diagnosticar:** por qué falló el descubrimiento de la GPU. El log no registra el estado de la GPU
  (`nvidia-smi`) ni si el servidor corría dentro de `jos_env`.

## vLLM Qwen3-14B: arranque del 19-nov-2026 (job 37415, compute-0-1, A100 40 GB completa)

**Objetivo:** repetir la línea base del 2-nov (mismos argumentos de `vllm serve`, pesos y venv en /home) con un único
cambio autorizado por el usuario: `VLLM_ENGINE_READY_TIMEOUT_S=3600` (el valor por defecto, 600 s, mató el arranque
del 2-nov después de cargar los pesos). Si el servidor quedaba listo (/health 200 y memoria GPU > 0), `medir_e1.sh`
lanzaba E1 (prefill) con `tesis/configs/e1_prefill_14b.toml` sin cambios. Scripts, logs y monitores en
`docs/arranque-14b-20261119/`.

**Medido**
- Reserva: `salloc` en la partición `gpu`, 8 CPU, 64 GB, límite 2 h. Empezó a las 06:46:57 y terminó a las 08:01:44
  con estado `COMPLETED` (1 h 14 min 47 s; no fue TIMEOUT). Qué cerró la reserva no consta en ningún log.
- `vllm serve` lanzado a las 07:29:31 (T0). Primer mensaje de vLLM en `serve.log` a las 07:55:19: **25 min 48 s**
  solo para importar Python/vLLM desde /home. El log termina a las 07:55:21 (arquitectura resuelta, `max_model_len`
  8192); no llegó a arrancar EngineCore ni a cargar pesos.
- GPU: 387 muestras de `nvidia-smi` cada 5 s (07:29:31 → 08:01:42), todas con **0 MiB y 0 %**.
- loadavg del nodo (1 min): 177,2 al empezar; entre 163,1 y 178,4 durante la prueba.
- `medir_e1.sh` nunca vio el servidor listo: `medir_e1.out` vacío y `resultados_14b/37415/` vacío. Open WebUI
  arrancó en :8090 pero avisó de que vLLM no respondía.

**Comparación del tramo T0 → primer mensaje de vLLM** (mismo comando, Qwen3-14B, venv en /home):

| fecha | job | nodo | loadavg al empezar | T0 → primer mensaje | desenlace |
|---|---|---|---|---|---|
| 29 oct | 31593 | compute-0-2 | 147,2 | 12 min 55 s | pesos cargados; el script lo detuvo a los 45 min |
| 2 nov | línea base | compute-0-1 | (no registrado) | 19 min 29 s | TimeoutError de vLLM a los 59 min 31 s |
| 19 nov | 37415 | compute-0-1 | 177,2 | 25 min 48 s | la reserva terminó a los 32 min 13 s |

**Sin diagnosticar:** por qué el tramo de importación pasa de 13 a 26 min entre pruebas. Lo único medido es que
el venv y los pesos se leen de /home por NFS en las tres. Ninguna prueba de vLLM con Qwen3-14B ha dado aún
métricas de inferencia.

## Gráficas

`graficas/` (regeneradas por `exportar_resumen.sh`; también en `~/tfg-graficas.zip`): 13 figuras en PDF y PNG,
con los números de cada una en `graficas/datos_graficas.json`. La prueba del 19-nov aparece en F09 (cronología de
arranques), F10 (memoria GPU) y F13 (loadavg).

## Pendiente
- Diagnosticar el `load_duration` por solicitud de Ollama (sale en los dos modelos, con loadavg 85 y 333).
- Repetir qwen3-coder:30b en las condiciones del 16-nov (servidor recién arrancado, contenedor `jos_env`, carga
  baja) para comparar los dos modelos en igualdad. El intento del 17-nov falló porque Ollama no detectó la GPU:
  antes de medir, comprobar en el log `offloaded 49/49 layers to GPU`.
- vLLM con Qwen3-14B: ningún arranque desde /home ha quedado listo (29-oct, 2-nov, 19-nov). Fase 2 (cronometrar el
  arranque desde /scratch) cuando el usuario lo decida.
