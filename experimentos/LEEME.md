# Gráficas del TFG — todas las pruebas hasta el 20-nov-2026

Cada figura está en `figuras/` en dos formatos: **PDF** (vectorial, para LaTeX/Word) y **PNG** a 300 ppp.
`datos_graficas.json` guarda cada número dibujado y el fichero del que sale; lo que es derivado
(calculado, no medido) está marcado. Regenerar todo:

```
graficas/.venv/bin/python graficas/generar_graficas.py
```

El script solo lee resultados; no modifica nada en `resultados/`, `ollama/resultados/` ni `tesis/`.

## Figuras

| fichero | qué muestra | origen |
|---|---|---|
| F01_ollama_desglose_ttft | TTFT del cliente separado en prefill + `load_duration` + resto, por longitud de prompt, para los dos modelos | `ollama/resultados/*/datos.json` |
| F02_ollama_prefill_toks | Velocidad de prefill (tok/s) frente a tokens del prompt | ídem |
| F03_ollama_tpot | TPOT (ms/token) frente a tokens del prompt | ídem |
| F04_ollama_carga_modelo | Tiempo de carga del modelo en GPU (incluye la corrida no válida del 11-nov, rayada) | ídem |
| F05_ollama_dispersion_ttft | Cada solicitud: TTFT del cliente frente a prefill del servidor | ídem |
| F06_vllm_piloto_mig_metricas | Piloto vLLM Qwen3-0.6B en MIG 5 GB: TTFT, TPOT y throughput, concurrencia 1 y 4 | `resultados/piloto-mig-0.6b-20261027-155013/` |
| F07_vllm_piloto_itl | Distribución de la latencia entre tokens del piloto | ídem |
| F08_vllm_8b_offline | Qwen3-8B en A100 (`vllm bench latency`): prefill a 128 y 512 tokens, decode derivado | `resultados/30732/` |
| F09_vllm_arranques_cronologia | Fases de los 4 arranques de `vllm serve` y en qué punto se quedó cada uno | logs de 31157, 31593, línea base 2-nov, 37415 |
| F10_vllm_arranques_memoria_gpu | Memoria GPU durante esos arranques | `gpu_monitor.csv` de cada uno |
| F11_vllm_8b_barrido_gpu | Memoria y utilización de GPU durante las 5 h del barrido 30732 | `resultados/30732/gpu_monitor.csv` |
| F12_preparacion_entorno_nfs | Tiempo de preparar el entorno / copiar a /scratch | `.out` y `logs/fase1_*.log` |
| F13_carga_nodo_loadavg | loadavg del nodo al empezar cada prueba | `entorno.txt` / `datos.json` |

## Qué queda fuera y por qué

- **Ollama nemotron-3-nano 11-nov**: solo aparece su tiempo de carga (F04): la medición no es válida (10 capas en CPU).
- **vLLM Qwen3-14B**: no hay métricas de inferencia; ningún arranque quedó listo (F09, F10).
- **Barrido 30732, in=512 out=129**: no terminó (TIMEOUT); no hay JSON.
