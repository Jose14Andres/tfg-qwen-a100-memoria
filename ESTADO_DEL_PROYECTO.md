# Estado del proyecto — TFG ITI-TFG-B26-0017

**Modelado experimental de las fases prefill y decode de Qwen3-8B en NVIDIA A100 mediante vLLM**

| Campo | Valor |
|---|---|
| Autor | José Andrés Ureta Granda |
| Tutor | Andrés Xavier Rubio Proaño |
| Institución | Universidad Tecnológica Indoamérica — Facultad de Ingenierías, Ingeniería en Tecnologías de la Información |
| Modalidad | Trabajo de Integración Curricular |
| Fecha de corte de este documento | 2026-10-24 |
| Repositorio remoto | `https://github.com/Jose14Andres/tfg-qwen-a100-memoria.git` |

Este documento registra todo lo trabajado hasta la fecha de corte: qué existe, en qué estado está y qué falta. La especificación completa del proyecto está en [`SPEC.md`](SPEC.md).

---

## 1. Resumen del proyecto

La inferencia de un LLM tiene dos fases con perfiles de cómputo distintos:

- **Prefill:** procesa todo el prompt en paralelo y produce el primer token. Limitada por cómputo; determina el **TTFT**.
- **Decode:** genera un token por paso leyendo pesos y caché KV. Limitada por ancho de banda de memoria; determina el **TPOT/ITL**.

El trabajo mide ambas fases de **Qwen3-8B** servido con **vLLM** en una **NVIDIA A100**, ajusta modelos predictivos de latencia y throughput en función de la longitud de entrada, la longitud de salida y la concurrencia, y los valida con datos *hold-out*.

**Objetivos específicos (SPEC §3.2):**

| # | Objetivo | Fase |
|---|---|---|
| OE1 | Analizar los fundamentos (prefill/decode, caché KV, PagedAttention, *continuous batching*) y derivar cotas roofline | Análisis |
| OE2 | Diseñar un experimento factorial que aísle cada fase | Diseño |
| OE3 | Implementar un banco de pruebas automatizado y reproducible | Implementación |
| OE4 | Ajustar y validar modelos de TTFT, TPOT y throughput (MAPE, R²) | Validación |

---

## 2. Estado por componente

| Componente | Ubicación | Estado |
|---|---|---|
| Especificación del proyecto | `SPEC.md` | Borrador v0.1 (2026-10-08), completo, con 7 decisiones abiertas |
| Plantilla y formato UTI | `uti_tfm.sty`, `main.tex` | Funcional; adaptado al nuevo formato institucional UTI (2026-10-22) y verificado contra la carpeta `Ingenieria_Tecnologias_Informacion_Nuevo_Formato/` (2026-10-24) |
| Páginas preliminares | `Preliminares/` | Estructura lista; lectores, resumen y abstract sin redactar |
| Capítulo I · Introducción | `Capítulos/01_Introducción.tex` | **Redactado** como ensayo argumentativo (~4 000 palabras), con las secciones de la plantilla UTI; falta la Fundamentación teórica y 1 `% FALTA CITA` (línea 16) |
| Capítulo II · Metodología | `Capítulos/02_Metodología.tex` | **Redactado** (2026-10-24) en las siete secciones de la plantilla; 8 `% FALTA CITA` pendientes |
| Capítulos III–IV | `Capítulos/03_*.tex`, `04_*.tex` | Texto guía y ejemplos de la plantilla; la línea 74 del Capítulo III cita `img:ejemplo_1` y `eq:disponibilidad`, eliminados del Capítulo II (referencias sin definir) |
| Referencias | `referencias_1.bib` (en uso) | 11 entradas verificadas; las 10 claves citadas resuelven. `referencias.bib` conserva las 6 entradas de ejemplo |
| Glosario y anexos | `Capítulos/06_Glosario.tex`, `Anexos/01_Anexo.tex` | Texto guía de la plantilla |
| Modelo de datos | `docs/modelo_datos/` | Documento completo (652 líneas, PDF generado) |
| Banco de pruebas | `experimentos/` | Base implementada (configuración, esquemas, prompts, entorno); 27 pruebas pasan |
| Manuales de la plantilla | `manual/` | Material de la plantilla institucional (uso y diseño) |
| Campaña experimental | — | No iniciada (falta acceso a la A100) |

---

## 3. Memoria en LaTeX

### 3.1 Estructura

```
main.tex                    Documento principal
uti_tfm.sty                 Formato institucional UTI
referencias_1.bib           Bibliografía en uso (IEEE, BibTeX)
referencias.bib             Entradas de ejemplo de la plantilla (no se usa)
Logos/                      LOGO-UTI-1..6.png
Preliminares/               01 Repositorio digital · 02 Aprobación tutor ·
                            03 Declaración de autenticidad · 04 Aprobación lectores ·
                            05 Dedicatoria · 06 Agradecimiento · 07 Índices ·
                            08 Resumen · 09 Abstract
Capítulos/                  01 Introducción · 02 Metodología · 03 Implementación ·
                            04 Conclusiones · 05 Referencias · 06 Glosario
Anexos/                     01 Anexo
```

### 3.2 Trabajo realizado en `main.tex`

- **Comentarios:** se reemplazaron los comentarios genéricos de la plantilla por los del proyecto: encabezado con título, código, autor y tutor; descripción de cada página preliminar; y resumen del contenido previsto de cada capítulo según el mapeo de SPEC §15.
- **Datos de portada:** título, facultad, carrera, modalidad, autor (único), tutor, ciudad y año ya están completos.
- **Título en inglés:** `Experimental Modeling of the Prefill and Decode Phases of Qwen3-8B on NVIDIA A100 Using vLLM` (traducción del título; por validar con el tutor).
- **Línea de investigación:** seleccionada `{4}{6}` — *Optimización del rendimiento* (línea 4). Alternativa: `{1}{3}` — *Sistemas de almacenamiento y computación en la nube*. Falta la confirmación del tutor.
- **Lectores:** siguen con el marcador `Nombres Apellidos`.
- **Tutor:** nombre corregido a Andrés Xavier Rubio Proaño.
- **Nuevo formato UTI (2026-10-22):** frase de título en portada según género (`\UTIFraseTituloPortada`), líneas horizontales alrededor del tema, preliminares sin sangría, modalidad «Trabajo de Integración Curricular previo a…», y fecha alineada a la derecha y «CERTIFICO» sin negrita en la aprobación del tutor y la declaración de autenticidad.
- **Citas:** estilo IEEE numérico (`IEEEtran`), confirmado por el tutor. `Capítulos/05_Referencias.tex` apunta a `referencias_1.bib`, aunque `CLAUDE.md` indica `referencias.bib`; hay que unificar.

### 3.3 Avance del Capítulo I

Reescrito el 2026-10-21 como ensayo argumentativo y reestructurado el 2026-10-24 según la plantilla UTI: Contextualización, Estado del Arte, Planteamiento del Problema (incluye la formulación de la pregunta), Justificación, Objetivos (General y Específicos), Alcance del Proyecto (Elementos incluidos y excluidos) y Fundamentación Teórica. La Fundamentación Teórica sigue sin redactar. `guia-capitulo-1_1.md` describe todavía la estructura anterior. Cita vLLM/PagedAttention, Orca, Sarathi-Serve, DistServe, FlashAttention, AWQ, Transformer, roofline, Qwen3 y la A100. Falta una fuente sobre la adopción de LLM en asistentes conversacionales y aplicaciones similares.

### 3.4 Contenido previsto por capítulo (SPEC §15)

| Capítulo | Contenido | Fuente en el SPEC |
|---|---|---|
| I · Introducción | Contextualización, problema, justificación, estado del arte (≥ 5 trabajos IEEE), objetivos, alcance, fundamentación teórica (roofline, MFU, MBU) | §1–4, §6 |
| II · Metodología | Diagnóstico, área de estudio, diseño factorial, técnicas e instrumentos, requerimientos RF/RNF, diseño del banco de pruebas, justificación técnica | §2, §5, §7, §8, §11, §12 |
| III · Implementación, pruebas y resultados | Banco de pruebas, campaña E1–E5, ajuste y validación de modelos, cronograma y presupuesto | §8–10, §13 |
| IV · Conclusiones | Conclusiones por objetivo, recomendaciones, trabajos futuros (cuantización, multi-GPU, desagregación prefill/decode, otros motores y modelos) | §15 |

### 3.5 Compilación

| Método | Comando | Resultado en este equipo |
|---|---|---|
| pdfLaTeX | `latexmk -pdf main.tex` | Compila sin errores; usa TeX Gyre Adventor como fuente sustituta |
| XeLaTeX | `latexmk -xelatex main.tex` | Falla: la fuente Century Gothic no está instalada |

El último `main.pdf` se generó con pdfLaTeX (MiKTeX) el 2026-10-24, sin citas sin definir, y se versiona en git. Quedan dos referencias cruzadas sin definir en el texto guía del Capítulo III. Si la universidad exige Century Gothic real, hay que instalar la fuente y compilar con XeLaTeX.

---

## 4. Modelo de datos (`docs/modelo_datos/Modelo_de_Datos.tex`)

Documento que define cómo se almacenan las mediciones. Se organiza en cuatro capas:

| Capa | Entidades | Origen |
|---|---|---|
| Contexto | `entorno`, `config_servidor`, `experimento` | Registradas al inicio de la corrida |
| Ejecución | `corrida`, `peticion`, `evento_token` | Cliente de carga |
| Telemetría | `muestra_gpu`, `muestra_vllm` | Muestreo a 100 ms |
| Modelado | `ajuste_modelo`, `coeficiente`, `ajuste_corrida`; vista `resumen_corrida` | Análisis |

La entidad central es `corrida`: un punto de la matriz experimental (combinación de L_in, L_out, B, λ y configuración) en una repetición concreta.

El documento incluye además: diagrama entidad–relación, diccionario de datos por tabla, métricas derivadas, reglas de integridad y calidad, almacenamiento físico (Parquet), flujo de datos y linaje, y ejemplos de registros.

**Decisiones abiertas del modelo de datos:**

- Umbrales SLO para el goodput.
- Si en E1 y E5 se guardan todos los `evento_token` o solo el primero y el último (por volumen).
- Periodo de muestreo de la telemetría (100 ms propuesto).
- Destino del respaldo de `data/raw`.

---

## 5. Banco de pruebas (`experimentos/`, paquete `tfgbench` 0.1.0)

### 5.1 Implementado

| Módulo | Función |
|---|---|
| `config.py` | Carga los YAML y expande la matriz experimental (producto cartesiano por barrido) |
| `schemas.py` | Esquemas Arrow/Parquet de cada tabla; única fuente de verdad del modelo físico |
| `ids.py` | Identificadores: hash de contenido para contexto, UUID v4 para ejecución |
| `promptgen.py` | Prompts sintéticos de longitud exacta como listas de IDs de token (RF2) |
| `entorno.py` | Captura del entorno de ejecución (tabla `entorno`) |
| `cli.py` | Comandos `tfgbench plan` y `tfgbench entorno` |

**Configuraciones de experimento (`experimentos/config/`):**

| Archivo | Experimento |
|---|---|
| `servidor_base.yaml` | Configuración común de vLLM |
| `piloto.yaml` | Corrida piloto: L_in {512, 4096} × L_out {1, 128} × B {1, 16} × chunked prefill on/off, 2 repeticiones |
| `e1_prefill.yaml` | E1 — Prefill aislado |
| `e2_decode.yaml` | E2 — Decode aislado |
| `e3_e2e.yaml` | E3 — Extremo a extremo |
| `e4_interferencia.yaml` | E4 — Interferencia prefill/decode |
| `e5_capacidad.yaml` | E5 — Capacidad y saturación |

**Pruebas:** 27 pruebas en `experimentos/tests/` (configuración, generación de prompts, esquemas e identificadores); todas pasan (verificado el 2026-10-24).

### 5.2 Uso

```bash
cd experimentos
python -m tfgbench plan config/piloto.yaml            # resume la matriz experimental
python -m tfgbench plan config/e1_prefill.yaml --csv matriz.csv
python -m tfgbench entorno --proveedor <nombre> --sin-gpu
python -m pytest                                       # pruebas
```

### 5.3 Pendiente (respecto a SPEC §11–12)

| Requerimiento | Falta |
|---|---|
| RF1 | Arranque y parada de vLLM con verificación de `/health` |
| RF3 | Clientes de carga *offline* (lotes fijos) y *online* (llegadas Poisson, streaming) |
| RF4 | Registro de tiempos por petición y por token |
| RF5 | Telemetría de GPU (`nvidia-smi`/DCGM) y métricas Prometheus de vLLM |
| RF6 | Orquestador (*runner*) que ejecute la matriz y persista las corridas |
| RF7 | Análisis: consolidación, ajuste de modelos, cotas roofline y figuras para LaTeX |

---

## 6. Control de versiones

| Rama | Commit | Contenido |
|---|---|---|
| `main` | `de488ce` | Incluye los PR #1 y #2 (Capítulo I) y los PR #3–#5 (nuevo formato UTI, citas y redacción del Capítulo II) |
| `docs/nuevo-formato-uti` | último commit | Fusionada en `main` hasta `c2c5298`; contiene además la reestructuración del Capítulo I según la plantilla UTI, **pendiente de subir y fusionar** |
| `docs/main-tex-datos-tesis` | `7e478af` | Ya fusionada en `main`; puede borrarse |

- Las ramas están sincronizadas con `origin`, salvo el último commit de `docs/nuevo-formato-uti`.
- Ya se versionan `SPEC.md`, `docs/modelo_datos/`, `experimentos/` (sin `.venv/`), `CLAUDE.md`, este archivo y `main.pdf`.
- Ignorados por `.gitignore`: auxiliares de LaTeX, `tmp_*`, `uploads/`, `prism-uploads/`, entorno y cachés de Python, e `informacion-tesis*.txt` (contiene correos de terceros).

---

## 7. Decisiones abiertas (SPEC §16)

1. Variante de la A100 (40/80 GB, PCIe/SXM) y proveedor de acceso.
2. Versión exacta de vLLM.
3. SLO para el goodput (umbrales de TTFT y TPOT).
4. Inclusión o no de un punto de comparación con cuantización.
5. Ubicación del código experimental (este repositorio o uno separado).
6. Línea/sublínea de investigación definitiva y título en inglés (ya hay valores propuestos en `main.tex`; falta la confirmación del tutor).
7. Metas de error definitivas (CA2–CA4).

---

## 8. Próximos pasos

1. Cerrar con el tutor las decisiones abiertas, en especial el acceso a la A100 y la línea de investigación.
2. Subir `docs/nuevo-formato-uti` y fusionar en `main` la reestructuración del Capítulo I.
3. Unificar la bibliografía: dejar un único `.bib` y alinear `CLAUDE.md` con `Capítulos/05_Referencias.tex`.
4. Resolver el `% FALTA CITA` del Capítulo I, redactar su Fundamentación Teórica y valorar si se añaden Splitwise y LLM-Viewer al estado del arte. Actualizar `guia-capitulo-1_1.md` a la estructura de la plantilla.
5. Resolver las 8 marcas `% FALTA CITA` del Capítulo II.
6. Completar el banco de pruebas (RF1, RF3–RF7) y ejecutar la corrida piloto.
7. Ejecutar la campaña E1–E5, ajustar y validar los modelos, y redactar los Capítulos III–IV.
8. Completar las páginas preliminares (lectores, resumen y abstract), el glosario y los anexos.
9. Actualizar el `README.md`, que solo documenta la compilación con `latexmk -pdf`.
