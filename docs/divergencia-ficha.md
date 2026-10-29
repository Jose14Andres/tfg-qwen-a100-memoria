# Divergencia entre los objetivos de la ficha aprobada y los de la memoria

**TFG:** ITI-TFG-B26-0017, «Modelado experimental de las fases prefill y decode de Qwen3-8B en NVIDIA A100 mediante vLLM»
**Autor:** José Andrés Ureta Granda
**Tutor:** Ing. Andrès Xavier Rubio Proaño
**Universidad Tecnológica Indoamérica**
**Fecha:** 29 de octubre de 2026

## Motivo del cambio

La ficha institucional aprobada formula un objetivo general con dos verbos («caracterizar y modelar») y cinco objetivos específicos. En la revisión del Capítulo 1 de la memoria, los objetivos se han reformulado con tres criterios:

1. Cada objetivo tiene **un solo verbo** en infinitivo.
2. Cada objetivo declara explícitamente **qué** se hace, **cómo** se hace y **para qué**.
3. El número de objetivos específicos baja de **cinco a cuatro**.

El contenido técnico del trabajo no cambia: se mantienen el modelo, el hardware, el motor de inferencia, los factores, los niveles experimentales y las métricas. Lo que cambia es cómo se agrupa ese contenido en objetivos.

## Objetivo general

| Ficha aprobada | Memoria (propuesta) |
|---|---|
| Caracterizar y modelar experimentalmente el comportamiento de las fases prefill y decode de Qwen3-8B sobre una GPU NVIDIA A100, evaluando el efecto de la cuantización, la longitud de contexto y la concurrencia sobre el rendimiento, la memoria, la energía y la calidad. | Identificar el efecto de la cuantización, la longitud de contexto y la concurrencia sobre el comportamiento de las fases prefill y decode de Qwen3-8B, servido mediante vLLM sobre una única GPU NVIDIA A100, en términos de rendimiento, memoria, energía y calidad, mediante la medición controlada de ambas fases y un modelo predictivo básico del TTFT y del TPOT, como base para la selección de configuraciones de inferencia según el escenario de uso. |

**Diferencias:**
- El verbo pasa de dos («caracterizar y modelar») a uno («identificar»).
- La caracterización y el modelado pasan a ser los medios del objetivo, es decir, su parte de «cómo».
- Se añade una finalidad explícita: servir de base para elegir configuraciones según el escenario de uso. Esa finalidad ya figuraba en la ficha, en la sección del producto tecnológico y en los escenarios de chat, consulta documental (RAG) y procesamiento por lotes.
- Se nombra vLLM y se precisa «una única GPU», que la ficha ya fijaba en la unidad de análisis C = (M, W, H, S).

## Objetivos específicos

| Ficha aprobada (5) | Memoria (4) | Relación |
|---|---|---|
| **OE1.** Implementar un entorno reproducible para ejecutar Qwen3-8B mediante vLLM sobre una A100. | **OE1.** Levantar un entorno reproducible de ejecución de Qwen3-8B mediante vLLM sobre una A100, mediante la fijación y documentación de las versiones de CUDA, del controlador de NVIDIA, de PyTorch y de vLLM en un contenedor o entorno versionado, como garantía de que las diferencias observadas entre configuraciones puedan atribuirse a los factores estudiados y no a cambios del entorno. | Se conserva. Cambia el verbo y se explicitan el cómo y el para qué, tomados de la fase 1 de la metodología de la ficha. |
| **OE2.** Diseñar un benchmark parametrizable para generar cargas con diferentes longitudes de entrada, salida y niveles de concurrencia. | **OE2.** Medir el TTFT, el TPOT, el throughput, la memoria GPU, la utilización y el consumo energético estimado, mediante un benchmark parametrizable con niveles controlados de longitud de entrada, longitud de salida y concurrencia sobre las representaciones BF16 y AWQ INT4, como base empírica del análisis por fases. | **Fusión:** el OE2 de la ficha (diseñar el benchmark) se integra en el nuevo OE2 como su instrumento, es decir, como su «cómo». |
| **OE3.** Medir TTFT, TPOT, throughput, memoria GPU, utilización y consumo energético. | *(integrado en el nuevo OE2)* | **Fusión:** el OE3 de la ficha (medir) pasa a ser el «qué» del nuevo OE2. |
| **OE4.** Analizar separadamente las fases prefill y decode e identificar sus principales cuellos de botella. | **OE3.** Analizar por separado el comportamiento de las fases prefill y decode, mediante el contraste de las mediciones con los límites de cómputo y de ancho de banda de memoria de la A100 según el modelo roofline, con vistas a la localización de los principales cuellos de botella de cada fase. | Se conserva y cambia de número (OE4 → OE3). Se explicita el método (modelo roofline). «Identificar» deja de ser un segundo verbo y pasa a la finalidad. |
| **OE5.** Construir un modelo predictivo básico para estimar TTFT y TPOT. | **OE4.** Construir un modelo predictivo básico del TTFT y del TPOT, a partir de las variables experimentales y de las relaciones identificadas en el análisis por fases, como soporte de la estimación y la recomendación de configuraciones del prototipo Qwen-A100 Profiler dentro de los niveles estudiados. | Se conserva y cambia de número (OE5 → OE4). La meta referencial de la ficha (MAPE inferior al 15–20 %, ajustable tras el piloto) se mantiene sin cambios en la memoria. |

## Contenido de la ficha que deja de estar cubierto por un objetivo

| Contenido de la ficha | Situación en la memoria |
|---|---|
| **El diseño del benchmark como resultado propio** (antiguo OE2). | El benchmark sigue existiendo y se nombra en el nuevo OE2, pero ya no es un objetivo con un criterio de cumplimiento propio. Su diseño (calentamiento, cinco repeticiones, orden aleatorizado de las configuraciones) queda documentado solo en el capítulo de Metodología. |
| **El verbo «modelar» en el objetivo general**, que coincide con el título del TFG («Modelado experimental…»). | El modelado se conserva como uno de los dos medios del objetivo general y como objetivo específico OE4, pero el verbo principal es ahora «identificar». El título y el verbo del objetivo general dejan de coincidir. |
| **La comparación limitada de calidad entre BF16 y AWQ INT4.** | Ningún objetivo específico la cubría tampoco en la ficha. Sigue cubierta solo por el objetivo general («calidad») y por el alcance del proyecto, igual que antes. Se señala aquí para que conste. |

## Efecto en el resto de la memoria

Al pasar de cinco a cuatro objetivos específicos, cambia la numeración. Hay que actualizar las referencias siguientes, que todavía no se han modificado:

- Capítulo 1, Justificación: «OE4» (análisis por fases) pasa a «OE3», y «OE5» (modelo predictivo) pasa a «OE4».
- Capítulo 1, Planteamiento: la «secuencia de productos» enumera cinco elementos; puede reducirse a cuatro para que coincida con los objetivos.
- Capítulo 2, Metodología: «tercer objetivo específico» (medición), en dos lugares, pasa a «segundo»; «cuarto objetivo específico» (análisis) pasa a «tercero». «Primer objetivo específico» no cambia.

---

Pendiente de confirmación por escrito del tutor, Ing. Andrès Xavier Rubio Proaño.
