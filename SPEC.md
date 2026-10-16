# SPEC — Modelado experimental de las fases prefill y decode de Qwen3-8B en NVIDIA A100 mediante vLLM

| Campo | Valor |
|---|---|
| Código institucional | ITI-TFG-B26-0017 |
| Autor | José Andrés Ureta Granda |
| Tutor | Andrès Xavier Rubio Proaño |
| Institución | Universidad Tecnológica Indoamérica — Facultad de Ingenierías, Ingeniería en Tecnologías de la Información |
| Modalidad | Trabajo de Integración Curricular |
| Línea / sublínea propuesta | 4.6 *Optimización del rendimiento* (alternativa: 1.3 *Sistemas de almacenamiento y computación en la nube*) — **por confirmar** |
| Estado del documento | Borrador v0.1 (2026-10-08) |

> Convención: lo marcado como **[DECIDIR]** es una decisión abierta que debe cerrarse con el tutor antes de ejecutar la campaña experimental. Lo marcado como **[SUPUESTO]** es una hipótesis de trabajo que debe verificarse.

---

## 1. Resumen

La inferencia de un modelo de lenguaje autorregresivo tiene dos fases con perfiles de cómputo distintos:

- **Prefill:** procesa todos los tokens del prompt en paralelo y genera el primer token. Es intensiva en cómputo (*compute-bound*) y determina el **TTFT** (*Time To First Token*).
- **Decode:** genera un token por paso y por secuencia, leyendo los pesos y la caché KV completos en cada paso. Está limitada por el ancho de banda de memoria (*memory-bound*) y determina el **TPOT/ITL** (*Time Per Output Token* / *Inter-Token Latency*).

El proyecto mide de forma controlada el comportamiento de ambas fases para **Qwen3-8B** servido con **vLLM** sobre una **NVIDIA A100**. A partir de esas mediciones construye **modelos analíticos/empíricos** que predicen la latencia y el throughput de cada fase según la longitud de entrada, la longitud de salida y la concurrencia, y valida su precisión con datos no usados en el ajuste.

## 2. Problema

- Dimensionar un servicio de LLM (cuántos usuarios concurrentes admite una GPU, qué latencia percibirán, cuánto cuesta cada token) suele hacerse por ensayo y error o con benchmarks agregados que mezclan prefill y decode.
- Las métricas agregadas (tokens/s totales) ocultan que las dos fases escalan de forma diferente: el prefill crece con la longitud del prompt (con un término cuadrático en la atención) y el decode con el tamaño de lote y la longitud de contexto acumulada (lecturas de la caché KV).
- Falta un modelo cuantitativo, reproducible y validado para una combinación concreta y relevante de modelo abierto, motor de servicio y GPU de centro de datos que permita predecir el rendimiento sin repetir campañas de benchmarking.

## 3. Objetivos

### 3.1 Objetivo general

Modelar experimentalmente el rendimiento de las fases prefill y decode de Qwen3-8B servido con vLLM en una GPU NVIDIA A100, mediante el diseño, la implementación y la validación de un banco de pruebas reproducible y de modelos predictivos de latencia y throughput.

### 3.2 Objetivos específicos

| # | Objetivo | Entregable verificable | Fase |
|---|---|---|---|
| OE1 | Analizar los fundamentos de la inferencia de transformers (prefill/decode, caché KV, PagedAttention, *continuous batching*) y las características de hardware de la A100 para derivar cotas teóricas (roofline). | Marco teórico + cálculo de cotas teóricas (sección 6) | Análisis |
| OE2 | Diseñar un experimento factorial que aísle cada fase, con variables, niveles, repeticiones y métricas definidas. | Matriz experimental (sección 8) | Diseño |
| OE3 | Implementar un banco de pruebas automatizado y reproducible (despliegue de vLLM, generador de carga, captura de métricas de GPU y almacenamiento de resultados). | Repositorio de código + datasets crudos | Implementación |
| OE4 | Ajustar modelos predictivos para TTFT, TPOT y throughput, y validarlos con datos *hold-out* usando métricas de error (MAPE, R²). | Modelos con coeficientes, error de validación ≤ meta (sección 10) | Validación |

## 4. Alcance

### 4.1 Incluye

- Un solo modelo: **Qwen3-8B** (pesos oficiales de Hugging Face, `Qwen/Qwen3-8B`) en precisión **BF16**.
- Un solo motor de servicio: **vLLM** (versión fijada; ver sección 7).
- Una sola GPU **NVIDIA A100** (sin paralelismo tensorial ni de pipeline).
- Cargas sintéticas controladas (prompts de longitud exacta y salida de longitud fija).
- Mediciones de latencia por fase, throughput, uso de memoria y utilización de GPU.
- Modelos de regresión/analíticos por fase y su validación.
- Análisis de dos configuraciones de planificación de vLLM: *chunked prefill* activado y desactivado.

### 4.2 Excluye

- Otros modelos, tamaños o familias (Qwen3-14B/32B, Llama, etc.), salvo una comparación menor opcional.
- Cuantización (AWQ, GPTQ, FP8) — se deja como trabajo futuro. **[DECIDIR]** si se incluye un punto de comparación.
- Multi-GPU, multinodo y arquitecturas desagregadas prefill/decode (p. ej. DistServe, Splitwise), que solo se discuten en el estado del arte.
- Evaluación de la calidad de las respuestas del modelo (exactitud, benchmarks de razonamiento).
- Otros motores (TensorRT-LLM, SGLang, TGI) como experimento; solo se tratan en el estado del arte.
- Desarrollo de una aplicación de usuario final.

## 5. Entorno experimental

### 5.1 Hardware

| Componente | Especificación | Nota |
|---|---|---|
| GPU | NVIDIA A100 **[DECIDIR: 40 GB o 80 GB; PCIe o SXM4]** | Determina el ancho de banda y la capacidad de la caché KV |
| Proveedor | **[DECIDIR: laboratorio universitario / nube (p. ej. Lambda, RunPod, GCP, AWS p4d)]** | Afecta al presupuesto y a la reproducibilidad |
| CPU / RAM | Registrar el modelo y la capacidad reales | El planificador de vLLM corre en la CPU |
| Almacenamiento | SSD/NVMe ≥ 100 GB | Pesos (~16 GB) + resultados |

Especificaciones de referencia de la A100 (hoja de datos de NVIDIA; **verificar con la variante usada**):

| Parámetro | A100 40 GB (SXM4) | A100 80 GB (SXM4) |
|---|---|---|
| Rendimiento BF16 Tensor Core (denso) | 312 TFLOPS | 312 TFLOPS |
| Ancho de banda HBM | ~1 555 GB/s | ~2 039 GB/s |
| Memoria | 40 GB HBM2 | 80 GB HBM2e |

### 5.2 Software

| Componente | Versión | Observación |
|---|---|---|
| SO | Ubuntu 22.04 LTS | |
| Driver NVIDIA / CUDA | Registrar (`nvidia-smi`) | |
| Python | 3.10–3.12 | Entorno aislado (`uv` o `venv`) |
| vLLM | **Fijar una versión exacta** (p. ej. `vllm==0.x.y`) | Congelar con `pip freeze` / lockfile |
| PyTorch | La que exija vLLM | |
| Monitoreo | `nvidia-smi --query-gpu`, DCGM (opcional), Nsight Systems (perfiles puntuales) | |
| Análisis | pandas, NumPy, SciPy, statsmodels / scikit-learn, matplotlib | |
| Contenedor (opcional) | Docker `vllm/vllm-openai:<tag>` | Mejora la reproducibilidad |

## 6. Fundamento teórico y cotas analíticas

### 6.1 Parámetros del modelo (Qwen3-8B)

Verificar contra el `config.json` oficial:

| Parámetro | Valor |
|---|---|
| Parámetros totales | ~8,2 B |
| Capas | 36 |
| Dimensión oculta | 4 096 |
| Cabezas de atención (Q / KV) | 32 / 8 (GQA) |
| Dimensión por cabeza | 128 |
| Contexto nativo | 32 768 tokens (hasta 131 072 con YaRN) |

### 6.2 Cálculos de referencia

- **Memoria de pesos (BF16):** ≈ 8,2·10⁹ × 2 B ≈ **16,4 GB**.
- **Caché KV por token:** 2 (K y V) × 36 capas × 8 cabezas KV × 128 × 2 B = **147 456 B ≈ 144 KiB/token**.
  - Con ~55 GB libres para KV en una A100 80 GB ≈ **~380 k tokens** en caché.
  - Con ~16 GB libres en una A100 40 GB ≈ **~110 k tokens**.
  - Esta capacidad limita la concurrencia máxima del decode.
- **FLOPs por token (lineales):** ≈ 2 × N_params ≈ 16,4 GFLOP/token, más el término de atención, que crece con la longitud del contexto.

### 6.3 Modelo roofline

- **Punto de equilibrio (*ridge point*) de la A100 80 GB:** 312·10¹² / 2,039·10¹² ≈ **153 FLOP/byte**.
- **Prefill** con L tokens: la intensidad aritmética ≈ L FLOP/byte (para L grande), así que con L ≳ 150 queda limitado por cómputo.
  - Cota inferior: T_prefill ≥ 2·N·L / FLOPS_pico.
- **Decode** con lote B: en cada paso se leen los pesos una vez (16,4 GB) más la caché KV de las B secuencias.
  - Cota inferior: T_paso ≥ (W + Σ KV_i) / BW.
  - Ejemplo en A100 80 GB con B pequeño: ≈ 16,4 GB / 2 039 GB/s ≈ **8 ms/paso** → ~125 tokens/s por secuencia como máximo teórico.

Las cotas sirven para expresar los resultados como **MFU** (*Model FLOPs Utilization*) en prefill y **MBU** (*Model Bandwidth Utilization*) en decode.

### 6.4 Conceptos de vLLM a cubrir en el marco teórico

PagedAttention, *continuous batching*, *chunked prefill*, *prefix caching*, CUDA graphs, `max_num_batched_tokens`, `max_num_seqs`, `gpu_memory_utilization` y preemption/swap de la caché KV.

## 7. Configuración base de vLLM

```bash
vllm serve Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --seed 0
# --no-enable-prefix-caching: evita reutilizar KV entre prompts
```

Lineamientos (ajustar las banderas a la sintaxis de la versión fijada):

- **Prefix caching desactivado**, para que cada prompt pague su prefill completo.
- **Chunked prefill** como factor experimental (on/off), con `max_num_batched_tokens` registrado.
- **Modo *thinking* de Qwen3:** irrelevante para la medición, porque la salida se fuerza a una longitud fija con `ignore_eos=True` y `max_tokens=N`. Aun así, se documenta.
- **Muestreo:** `temperature=0` (greedy) para que el resultado sea determinista.
- **Calentamiento (*warm-up*):** al menos 5 peticiones descartadas antes de cada serie.

## 8. Diseño experimental

### 8.1 Variables independientes

| Factor | Símbolo | Niveles propuestos |
|---|---|---|
| Longitud de entrada (tokens) | L_in | 128, 256, 512, 1 024, 2 048, 4 096, 8 192, 16 384 |
| Longitud de salida (tokens) | L_out | 1 (prefill puro), 32, 128, 512, 1 024 |
| Concurrencia / tamaño de lote | B | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| Chunked prefill | CP | on / off |
| Tasa de llegada (modo servidor) | λ | Poisson; 0,5–20 req/s (barrido hasta saturar) |

Cada combinación se ejecuta **≥ 5 repeticiones**. Se reporta la media, la mediana, p90 y p99 con intervalos de confianza del 95 %. Las combinaciones que no caben en la caché KV se registran como límite de capacidad.

### 8.2 Experimentos

| ID | Objetivo | Diseño | Métrica principal |
|---|---|---|---|
| E1 — Prefill aislado | Modelar TTFT vs L_in y B | L_out = 1, barrido L_in × B, modo offline | TTFT, tokens de prefill/s, MFU |
| E2 — Decode aislado | Modelar TPOT vs B y contexto | L_in fijo pequeño (128), L_out = 1 024, barrido B; luego L_in variable con B fijo | TPOT/ITL, tokens/s, MBU |
| E3 — Extremo a extremo | Validar la composición E2E ≈ TTFT + (L_out − 1)·TPOT | Combinaciones mixtas L_in × L_out × B | Latencia E2E, error de predicción |
| E4 — Interferencia | Medir cómo el prefill de peticiones nuevas penaliza el decode de las activas (CP on/off) | Carga en línea con llegadas Poisson | ITL p99, TTFT p99, goodput |
| E5 — Capacidad | Encontrar la concurrencia máxima sin preemption y el punto de saturación | Barrido de λ hasta la saturación | Throughput máx., λ de saturación |

### 8.3 Métricas

| Métrica | Definición | Fuente |
|---|---|---|
| TTFT | Tiempo desde el envío hasta el primer token | Cliente (streaming) |
| TPOT / ITL | (E2E − TTFT) / (L_out − 1); ITL = intervalos entre tokens | Cliente |
| Latencia E2E | Envío → último token | Cliente |
| Throughput | Tokens de salida/s y peticiones/s | Cliente + logs de vLLM |
| Goodput | Peticiones/s que cumplen un SLO (p. ej. TTFT < 1 s, TPOT < 50 ms) **[DECIDIR SLO]** | Derivada |
| Memoria GPU, utilización SM, potencia | Muestreo a 100 ms | `nvidia-smi` / DCGM |
| Uso de caché KV, preemptions | Métricas Prometheus de vLLM (`/metrics`) | Servidor |
| MFU / MBU | Medido / pico teórico (sección 6) | Derivada |

## 9. Modelos a construir

Forma funcional inicial; la estructura final se elige por ajuste y parsimonia.

**Prefill** (lote de B prompts de longitud L):

$$T_{prefill}(B, L) = a_0 + a_1 \cdot B L + a_2 \cdot B L^2$$

El término lineal corresponde a las capas densas y el cuadrático a la atención.

**Decode** (paso con B secuencias y contexto medio C):

$$T_{paso}(B, C) = b_0 + b_1 \cdot B + b_2 \cdot B C$$

b₀ ≈ lectura de pesos / BW; b₂ ≈ lectura de la caché KV.

**Extremo a extremo:**

$$T_{E2E} \approx T_{prefill} + \sum_{k=1}^{L_{out}-1} T_{paso}(B_k, C_k)$$

**Throughput de decode:** tokens/s = B / T_paso(B, C), con saturación por la capacidad de la caché KV.

**Procedimiento:**

1. Ajuste por mínimos cuadrados (OLS/NNLS) con el 70–80 % de los puntos.
2. Validación con el 20–30 % *hold-out* y con E3/E4.
3. Comparación contra las cotas roofline de la sección 6.
4. Análisis de residuos y de puntos de quiebre (cambio de régimen memory-bound → compute-bound en el decode con B grande).

## 10. Criterios de aceptación

| ID | Criterio | Meta |
|---|---|---|
| CA1 | Banco de pruebas reproducible: un comando ejecuta la matriz completa y genera CSV/Parquet | Re-ejecución con variación de la mediana ≤ 5 % |
| CA2 | Error del modelo de TTFT en hold-out | MAPE ≤ 10 %, R² ≥ 0,95 **[SUPUESTO; ajustar con el tutor]** |
| CA3 | Error del modelo de TPOT en hold-out | MAPE ≤ 10 %, R² ≥ 0,95 |
| CA4 | Predicción E2E (E3) | MAPE ≤ 15 % |
| CA5 | Resultados contrastados con las cotas teóricas | MFU y MBU reportados por régimen |
| CA6 | Datos crudos, código y entorno versionados y publicados (anexo) | Enlace al repositorio en el Anexo |

## 11. Requerimientos del banco de pruebas

### 11.1 Funcionales

- **RF1:** Desplegar vLLM con una configuración parametrizada (YAML) y verificar su salud (`/health`).
- **RF2:** Generar prompts sintéticos de longitud exacta en tokens usando el tokenizer de Qwen3.
- **RF3:** Ejecutar la carga en modo *offline* (lote fijo) y en modo *online* (llegadas Poisson, cliente asíncrono con streaming).
- **RF4:** Registrar los tiempos por petición y por token (timestamps de alta resolución en el cliente).
- **RF5:** Muestrear en paralelo la telemetría de la GPU y las métricas Prometheus de vLLM.
- **RF6:** Persistir cada corrida con metadatos (versiones, hardware, configuración, semilla, timestamp).
- **RF7:** Ajustar los modelos y generar automáticamente las tablas y figuras para LaTeX.

### 11.2 No funcionales

- **RNF1 — Reproducibilidad:** versiones fijadas, semillas fijas, contenedor o lockfile, scripts idempotentes.
- **RNF2 — Aislamiento:** GPU dedicada sin otros procesos durante las mediciones (verificado con `nvidia-smi`).
- **RNF3 — Precisión de medición:** reloj monotónico (`time.perf_counter_ns`), warm-up descartado y repeticiones suficientes.
- **RNF4 — Trazabilidad:** cada figura o tabla de la memoria referencia el ID de la corrida de origen.
- **RNF5 — Costo:** la campaña completa cabe en el presupuesto de horas-GPU (sección 13).

## 12. Arquitectura del banco de pruebas

```
config/                 # YAML de experimentos (factores y niveles)
bench/
  server.py             # arranque/parada de vLLM con config dada
  promptgen.py          # prompts de longitud exacta (tokenizer Qwen3)
  client_offline.py     # API LLM de vLLM, lotes fijos
  client_online.py      # cliente async OpenAI-compatible, streaming, Poisson
  telemetry.py          # nvidia-smi / DCGM + scrape /metrics
  runner.py             # orquesta la matriz, repeticiones y warm-up
analysis/
  load.py               # consolida resultados → Parquet
  fit.py                # ajuste de modelos y validación
  roofline.py           # cotas teóricas, MFU/MBU
  figures.py            # genera PDF/PNG para Capítulos/fig/
data/raw/  data/processed/
```

Flujo: `config YAML → runner → (server + client + telemetry) → data/raw → analysis → figuras/tablas → memoria LaTeX`.

**[DECIDIR]** Si el código vive en este mismo repositorio (carpeta `experimentos/`) o en uno separado enlazado desde el Anexo.

## 13. Cronograma (borrador)

| Semana | Actividad | Salida |
|---|---|---|
| 1–2 | Revisión bibliográfica, marco teórico y cotas teóricas | Cap. I (borrador), `referencias.bib` |
| 3 | Gestión del acceso a la GPU, montaje del entorno y prueba de humo de vLLM | Entorno documentado |
| 4–5 | Implementación del banco de pruebas (RF1–RF6) | Código + corrida piloto |
| 6 | Experimento piloto, ajuste de niveles y estimación del tiempo de campaña | Matriz final |
| 7–8 | Campaña experimental E1–E5 | Datos crudos |
| 9–10 | Ajuste y validación de modelos, figuras | Cap. III (resultados) |
| 11–12 | Redacción de los capítulos II–IV, revisión con el tutor | Memoria completa |
| 13 | Correcciones, resumen/abstract y preparación de la defensa | Versión final |

**Presupuesto [DECIDIR]:** horas-GPU estimadas tras el piloto. Como referencia, una A100 en la nube cuesta aproximadamente USD 1–4/h según el proveedor; la campaña completa se estima en 40–80 h, que hay que confirmar.

## 14. Riesgos

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Sin acceso estable a la A100 | Alto | Confirmar el proveedor en la semana 3; reservar presupuesto en la nube como respaldo |
| Cambios de API/comportamiento entre versiones de vLLM | Medio | Fijar la versión; documentar las banderas exactas |
| Variabilidad de las mediciones (ruido térmico, *clock boost*, vecinos en la nube) | Medio | Repeticiones, mediana/percentiles, registrar los relojes de la GPU; instancia dedicada |
| La A100 de 40 GB limita el contexto o el lote | Medio | Ajustar los niveles de L_in y B a la capacidad de la caché KV calculada |
| Modelos con error alto en la zona de saturación | Medio | Modelos por tramos; reportar el rango de validez |
| El tiempo de campaña excede lo previsto | Medio | Piloto en la semana 6; diseño fraccional si hace falta |

## 15. Mapeo a la memoria (formato PIC UTI)

| Capítulo / sección | Contenido del spec |
|---|---|
| I · Contextualización / Problema / Justificación | Secciones 1–2 |
| I · Estado del arte (≥ 5 trabajos IEEE) | vLLM/PagedAttention (Kwon et al., 2023), Orca (Yu et al., 2022), Sarathi-Serve (Agrawal et al., 2024), DistServe (Zhong et al., 2024), Splitwise (Patel et al., 2024), LLM-Viewer/roofline (Yuan et al., 2024), Qwen3 Technical Report (2025) — **verificar las referencias exactas** |
| I · Objetivos / Alcance | Secciones 3–4 |
| I · Fundamentación teórica | Sección 6 |
| II · Diagnóstico / Área de estudio | Secciones 2 y 5 |
| II · Metodología | Sección 8 (enfoque experimental cuantitativo, diseño factorial) |
| II · Requerimientos / Diseño / Justificación técnica | Secciones 7, 11, 12 |
| III · Implementación, pruebas y resultados | Secciones 8–10 (ejecución) |
| III · Cronograma y presupuesto | Sección 13 |
| IV · Conclusiones / Trabajos futuros | Cuantización FP8/AWQ, multi-GPU, desagregación prefill/decode, otros motores y modelos |

## 16. Decisiones abiertas

1. Variante de la A100 (40/80 GB, PCIe/SXM) y proveedor de acceso.
2. Versión exacta de vLLM.
3. SLO para el goodput (umbrales de TTFT y TPOT).
4. Inclusión o no de un punto de comparación con cuantización.
5. Ubicación del código experimental (este repositorio o uno separado).
6. Línea/sublínea de investigación definitiva y título en inglés.
7. Metas de error definitivas (CA2–CA4).
