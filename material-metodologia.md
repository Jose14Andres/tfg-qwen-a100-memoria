# Material para el capítulo de Metodología

TFG ITI-TFG-B26-0017 · José Andrés Ureta Granda
Destilado de `PRUEBA_DE_HUMO.md`, `salloc_gpu.md`, `bench_pd.sh`, `resumen_pd.py` y los trabajos 30457, 30724 y 30732.
Destino sugerido: `docs/material-metodologia.md`.

---

## 0. Advertencia previa: qué NO es prescindible

El trabajo sobre los nodos no es andamiaje que se tira después de usarlo. Tres de esos
resultados **son** contenido de la memoria, y uno de ellos es probablemente lo más
original que has producido:

1. **El parche de compatibilidad glibc.** Diagnosticaste que una rueda etiquetada
   `manylinux_2_31` exige en realidad símbolos de glibc 2.34, identificaste los cuatro
   símbolos concretos, y los resolviste sin recompilar ni contenerizar. Eso es una
   contribución técnica real y afecta directamente a la reproducibilidad del entorno.
2. **Las restricciones operativas del clúster.** Un solo trabajo simultáneo, máximo
   2 GPUs en `gpu`, 1 GPU en `gpu-dev`, nodos compartidos con carga variable. Estas
   restricciones **determinaron** el diseño experimental. Un capítulo de metodología que
   no las declare parece arbitrario.
3. **La contención medida.** El load average de 317 sobre 256 núcleos del 22-oct es un
   dato que condiciona la validez de toda medición de latencia.

Lo que sí sobra de la memoria: la narración cronológica de los intentos fallidos de
instalación. Eso va a un anexo o a la bitácora, no al cuerpo.

---

## 1. Plataforma experimental

| Elemento | Valor |
|---|---|
| Clúster | HPC CEDIA (`hpc.cedia.edu.ec`), planificador Slurm |
| Nodos de cómputo GPU | `compute-0-1`, `compute-0-2`: 8× NVIDIA A100-SXM4-40GB cada uno |
| Nodo interactivo | `compute-0-0`: 1× A100-40GB + 6× MIG `a100_2g.10gb` + 2× MIG `a100_1g.5gb` |
| CPU por nodo | 256 núcleos |
| RAM por nodo | ~960 GB (nodos GPU), ~1 TB (interactivo) |
| Memoria GPU | 40 960 MiB |
| Límite de potencia | 400 W |
| Ancho de banda HBM2 | 1555 GB/s |
| Rendimiento BF16 Tensor Core | 312 TFLOPS (denso) |
| Sistema operativo | Ubuntu 20.04.6, glibc 2.31 |
| Almacenamiento de trabajo | NFS compartido (`$HOME`) |

### Advertencia sobre el driver

| Fecha | Nodo | Driver |
|---|---|---|
| 20-oct | `compute-0-1` | 535.161.08 |
| 22-oct | `compute-0-2` | 575.51.03 |
| 24-oct | `compute-0-1` y `compute-0-2` | 575.51.03 |

**El driver de `compute-0-1` cambió durante la semana de preparación.** No es una anécdota:
significa que el clúster actualiza drivers sin aviso, y que dos mediciones separadas por
días pueden no ser comparables.

Consecuencias que la memoria debe declarar:
- Toda ejecución registra el driver en el momento de medir (`bench_pd.sh` ya lo hace en
  `entorno.txt`).
- La campaña definitiva debe completarse dentro de una ventana sin actualizaciones, o
  bien reportar el driver como covariable.

### Advertencia sobre MIG

En un trabajo con un trozo MIG, `nvidia-smi` muestra la A100 padre de 40 GB, no la
rebanada real. Para identificar el dispositivo efectivo hay que usar `nvidia-smi -L`,
`torch.cuda.get_device_name()` o `torch.cuda.mem_get_info()`.

Las particiones MIG quedan excluidas de la campaña de medición por dos razones: la memoria
es insuficiente para Qwen3-8B, y NVML no expone potencia por instancia MIG, lo que
impide atribuir el consumo energético exigido por el objetivo específico 3.

---

## 2. Entorno de software congelado

| Componente | Versión |
|---|---|
| Python | 3.11 (módulo `python/3.11`, miniconda) |
| PyTorch | 2.11.0+cu129 |
| CUDA runtime | 12.9 |
| vLLM | 0.20.0+cu129 |
| Procedencia de la rueda | `wheels.vllm.ai`, commit `88d34c6409e9fb3c7b8ca0c04756f061d2099eb1` |
| Backend de atención | FlashAttention 2 |
| Modelo | `Qwen/Qwen3-8B`, revisión `b968826d9c46dd6066d109eabc6255188de91218` |
| Caché de modelos | `HF_HOME=~/tfg-jose-ureta/hf-cache` |
| Dependencias completas | `req-congelado.txt` |

### Justificación de fijar vLLM 0.20.0

Las versiones posteriores de vLLM distribuyen su stack compilado contra **CUDA 13**, que
requiere driver ≥ 580. El clúster operaba con driver 535 en el momento de la instalación.
La compatibilidad de versión menor de CUDA cubre runtimes 12.x sobre drivers ≥ 525, pero
no cruza de la versión mayor 12 a la 13.

Esta es la redacción honesta: la versión no se eligió por preferencia, sino por la
restricción del entorno, y queda documentada como tal.

### FlashAttention 2 y no 3

vLLM selecciona FlashAttention 2 porque la versión 3 solo está implementada para la
arquitectura Hopper (H100). La A100 es Ampere. Es una limitación de la plataforma, no una
decisión del experimento, y conviene declararla porque afecta al rendimiento absoluto de
la fase de prefill.

---

## 3. Parche de compatibilidad glibc

### El problema

La rueda `vllm-0.20.0+cu129` se distribuye con la etiqueta `manylinux_2_31`, que declara
compatibilidad con glibc 2.31. Sin embargo, dos de sus extensiones nativas —`_C.abi3.so`
y `_moe_C.abi3.so`— enlazan cuatro símbolos con versión `GLIBC_2.34`:

```
dlopen, dlclose, dlsym, pthread_once
```

Los nodos del clúster ejecutan Ubuntu 20.04.6 con glibc 2.31. El resultado es un
`ImportError` al cargar vLLM. **La etiqueta de la rueda es incorrecta.**

### El diagnóstico

Se inspeccionaron con `objdump -T` todas las bibliotecas compartidas del entorno virtual.
Solo cuatro declaran dependencias de glibc 2.32 o superior:

| Biblioteca | ¿Se carga en este clúster? |
|---|---|
| `vllm/_C.abi3.so` | Sí — núcleo de vLLM |
| `vllm/_moe_C.abi3.so` | Sí |
| `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so` | No — FlashAttention 3 es solo Hopper |
| `vllm/third_party/deep_gemm/_C.cpython-312-*.so` | No — compilada para Python 3.12 |

Los cuatro símbolos problemáticos no corresponden a funcionalidad ausente. En glibc 2.34
se consolidaron `libdl` y `libpthread` dentro de `libc`; en glibc 2.31 esos mismos
símbolos existen, pero residen en `libdl.so.2` y `libpthread.so.0`. Es un artefacto de
haber compilado la rueda sobre un sistema más moderno.

### La solución

`parche_glibc.sh` aplica tres operaciones:

1. **Eliminar la versión de los símbolos** con `patchelf --clear-symbol-version` y añadir
   `libdl.so.2` y `libpthread.so.0` como dependencias explícitas (`--add-needed`).
2. **Marcar como `VER_FLG_WEAK`** la entrada `GLIBC_2.34` en la sección `.gnu.version_r`
   (`parche_glibc_weak.py`). El paso anterior por sí solo no basta: el cargador dinámico
   sigue rechazando la biblioteca mientras esa entrada exista sin la bandera.
3. **Verificar** que `import vllm._C` y `import vllm._moe_C` funcionen.

Propiedades: idempotente, conserva los originales como `*.so.orig`, y debe reaplicarse
tras cada reinstalación del entorno virtual. `bench_pd.sh` lo invoca automáticamente si
detecta que vLLM no importa.

### Alternativas evaluadas y descartadas

| Alternativa | Motivo del descarte |
|---|---|
| Contenedor oficial `vllm/vllm-openai` con enroot/pyxis o Apptainer | Imagen superior a 10 GB sobre NFS compartido. Queda como plan B |
| Compilar vLLM desde el código fuente en el nodo | Varias horas de compilación por reinstalación |
| Fijar una versión anterior de vLLM | No resolvía el problema: la etiqueta incorrecta aparece en varias versiones |

### Cómo redactarlo

Dos puntos de honestidad que un lector técnico va a valorar:

- El parche **no cambia el comportamiento numérico** de vLLM: solo reubica dónde el
  cargador dinámico busca cuatro símbolos. No altera kernels, precisión ni planificación.
- Es una **modificación al binario distribuido**, y por tanto el entorno no es
  bit-a-bit idéntico al que se obtendría con `pip install vllm==0.20.0` en un sistema con
  glibc ≥ 2.34. La memoria debe decirlo, y el script debe acompañar al repositorio para
  que cualquiera reproduzca exactamente el mismo entorno.

---

## 4. Restricciones operativas que condicionaron el diseño

### Límites de Slurm

| Partición | Tiempo máx. | GPUs máx. | CPUs máx. | RAM máx. |
|---|---|---|---|---|
| `interactive` *(por defecto)* | 1 día | — | — | — |
| `gpu` | 2 días | 2 | 160 | 128 G |
| `gpu-max` | 1 día | 4 | 128 | 256 G |
| `gpu-dev` | 4 días | 1 | 16 | — |

- La QoS `normal` del usuario permite **un único trabajo en ejecución simultánea**.
- **No es posible reservar un nodo completo**: el máximo son 4 GPUs en `gpu-max`.
- La partición por defecto es `interactive`, que solo ofrece particiones MIG. Omitir
  `-p gpu` conduce a un recurso inservible para este experimento.

### Consecuencias directas sobre el diseño

1. **Ejecución secuencial obligatoria.** Las configuraciones no pueden paralelizarse.
2. **Elección de `gpu-dev` sobre `gpu`.** El 24-oct, `sbatch --test-only` en `gpu`
   estimaba el inicio para el 31-oct pese a haber GPUs libres; en `gpu-dev` la estimación
   era inmediata. `gpu-dev` ofrece además 4 días de walltime frente a 2.
3. **Reutilización del motor.** Arrancar vLLM tiene un costo fijo de decenas de segundos
   (más de 80 s en el peor caso observado, con el entorno virtual sobre NFS). Reiniciar el
   motor por configuración sería prohibitivo, de modo que la fase en línea levanta el
   servidor una vez y recorre las configuraciones contra él.

### Contención del nodo

Los nodos son compartidos. El 22-oct a las 19:34 se midió en `compute-0-2`:

```
load average: 317,33 / 337,72 / 214,73   sobre 256 núcleos
```

Es decir, 1,24× sobresuscrito en la ventana de un minuto y 1,32× en la de cinco, frente a
0,84× quince minutos antes. **La carga se duplicó durante la ejecución.**

Dado que TTFT y TPOT dependen de que la CPU alimente a la GPU sin demora, la contención
del nodo es una variable de confusión. Medidas adoptadas:

- Registro de `/proc/loadavg` al inicio de cada trabajo (`entorno.txt`).
- Monitor de `nvidia-smi` cada 2 s durante toda la ejecución (`gpu_monitor.csv`).
- Repeticiones suficientes para estimar la varianza y detectar ejecuciones contaminadas.

---

## 5. Diseño de medición implementado

`bench_pd.sh` organiza la ejecución en cinco fases:

| Fase | Contenido | Salidas en `resultados/<jobid>/` |
|---|---|---|
| 0 | Nodo, GPU, versiones, carga; lanza el monitor de telemetría | `entorno.txt`, `gpu_monitor.csv` |
| 1 | Prueba de humo con el modelo objetivo | `smoke.txt`, `smoke.err` |
| 2 | Barrido sin servidor (`vllm bench latency`) | `lat_in*_out*_bs*.json` |
| 3 | Barrido en línea (`vllm serve` + `vllm bench serve`) | `serve_in*_out*_c*.json`, `metrics_servidor.txt` |
| 4 | Agregación y tablas | `RESUMEN.md` |

### Triangulación de la separación prefill/decode

Este es el núcleo metodológico del trabajo, y conviene presentarlo como lo que es: **tres
métodos independientes que miden lo mismo**, de modo que sus discrepancias acotan el error.

**Método 1 — Diferencia de latencias sin servidor.**
Con `output-len = 1`, el tiempo medido es esencialmente prefill. Con `output-len = 129`,
se añaden 128 pasos de decode. De ahí:

```
T_prefill        ≈ T(out = 1)
throughput_prefill = entrada × lote / T(out = 1)
TPOT             = [ T(out = 129) − T(out = 1) ] / 128
```

Ventaja: aísla las fases sin instrumentación interna. Limitación: `T(out=1)` incluye una
pequeña fracción de decode y el sobrecoste de arranque de la iteración.

**Método 2 — Métricas de cliente en línea.**
`vllm bench serve` con `--ignore-eos` reporta TTFT, TPOT e ITL con percentiles 50, 90
y 99. Es la perspectiva del usuario e incluye tiempo de cola.

**Método 3 — Métricas internas del servidor.**
El endpoint `/metrics` expone `request_prefill_time_seconds` y
`request_decode_time_seconds`, que separan las fases según la contabilidad interna del
motor. Es la medida más directa y sirve de referencia para validar los métodos 1 y 2.

> En la memoria, presentar las tres y comparar. Si convergen, la medición es robusta. Si
> no, la discrepancia es en sí misma un resultado que merece discusión.

### Techos teóricos de referencia

Para Qwen3-8B en BF16 (≈ 8,2·10⁹ parámetros, ≈ 16,4 GB de pesos) sobre A100-SXM4-40GB:

```
Decode, lote 1   : 1555 GB/s ÷ 16,4 GB        ≈    95 tok/s   (limitado por memoria)
Prefill          : 312 TFLOPS ÷ (2 × 8,2·10⁹) ≈ 19 000 tok/s  (limitado por cómputo)
```

Dos órdenes de magnitud de diferencia. Contrastar las mediciones contra estos techos es lo
que convierte números sueltos en evidencia del argumento central del trabajo.

### Presupuesto de memoria verificado

```
Capas: 36 · Cabezas KV: 8 · Dimensión de cabeza: 128
KV por token = 2 × 36 × 8 × 128 × 2 bytes = 147 456 B = 144,0 KiB

8192 tokens × 16 secuencias → 18,0 GiB de caché KV
Pesos BF16                 → 15,3 GiB
Total                      → 33,3 GiB de ~36 GiB utilizables
```

Valores obtenidos del `config.json` de la revisión del modelo efectivamente descargada,
no de la documentación.

---

## 6. Divergencias respecto a la ficha aprobada — RECONCILIAR

La matriz declarada en la ficha institucional no coincide con la implementada en
`bench_pd.sh`:

| Variable | Ficha aprobada | `bench_pd.sh` |
|---|---|---|
| Cuantización | BF16 **y AWQ INT4** | solo BF16 |
| Longitud de entrada | 128, 512, 2048, **8192** | 128, 512, 1024, 2048, 4096 |
| Longitud de salida | 64, 256 | 1, 129 *(offline)*; 128, 256 *(online)* |
| Concurrencia | 1, 4, 8, 16 | 1, 2, 4, 8, 16, 32, 64 |
| Repeticiones | 5 | 10 iteraciones *(offline)* |

Ninguna divergencia es necesariamente un problema, pero **todas deben declararse**. Dos
salidas posibles:

1. **Declararlo como piloto.** Es lo que la ficha prevé: *"Esta meta deberá ajustarse
   después de los experimentos piloto"*. El barrido actual explora el espacio para
   calibrar la matriz definitiva.
2. **Reconciliar la matriz final** con la ficha, o acordar por escrito con el tutor la
   matriz revisada antes de la campaña.

### Dos huecos que importan

**No se mide el escenario de 8192 tokens.** Es justamente la configuración límite: 33,3 GiB
de los ~36 disponibles. Es la que decide si la matriz de la ficha es ejecutable, y la que
puede obligar a `--enforce-eager` por los grafos CUDA. Debería estar en el piloto.

**No se mide AWQ INT4.** Es la mitad de la comparación que exige el objetivo general. El
riesgo asociado —disponibilidad y fiabilidad del checkpoint cuantizado— sigue abierto y
debería cerrarse pronto.

---

## 7. Lo que falta para cerrar el capítulo

- [ ] Resultados del trabajo 30732 con Qwen3-8B en A100 completa
- [ ] Prueba del escenario límite: 8192 tokens, concurrencia 16, BF16
- [ ] Identificación y validación de un checkpoint AWQ INT4
- [ ] Matriz definitiva acordada por escrito con el tutor
- [ ] Protocolo de energía: ventana de integración y medición del idle por campaña
