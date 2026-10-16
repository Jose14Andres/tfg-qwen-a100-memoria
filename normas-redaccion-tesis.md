# Normas de redacción — TFG ITI-TFG-B26-0017

**Modelado experimental de las fases prefill y decode de Qwen3-8B en NVIDIA A100 mediante vLLM**
José Andrés Ureta Granda · Universidad Tecnológica Indoamérica

Documento de trabajo. Destino sugerido: `docs/normas-redaccion.md` en el repositorio de la memoria.

---

## 0. Qué autoridad tiene cada documento

Antes de aplicar nada, conviene tener claro el orden de prelación. No todos los documentos que estás usando mandan lo mismo:

| Documento | Qué gobierna | Autoridad |
|---|---|---|
| Ficha ITI-TFG-B26-0017 | Título, objetivos, alcance, metodología, entregables | **Normativa.** Es el acta aprobada |
| Plantilla `uti_tfm.sty` + estructura de `Capitulos/` | Formato, márgenes, tipografía, estructura de capítulos, estilo de citación | **Normativa.** Es el formato institucional |
| Indicaciones del Ing. Rubio | Todo lo anterior, en caso de duda | **Normativa.** Decide él |
| Soriano, *Cómo se escribe una tesis* (2008) | Oficio de redacción: lógica interna, estilo, errores frecuentes | **Orientativa.** Es un manual comercial, no una norma de la UTI |

El libro de Soriano es útil, y abajo destilo lo que sí aplica. Pero está escrito para tesis doctorales españolas de humanidades y derecho, en 2008. Hay partes que, si las sigues al pie de la letra, te alejan de lo que tu facultad espera de un trabajo experimental de ingeniería.

---

## 1. Lo que del manual sí aplica a tu caso

### 1.1 Numeración decimal, no romana

Soriano distingue dos diseños: el tradicional (I, A, 1) y el moderno, de numeración decimal escalonada, que señala explícitamente como el propio de los trabajos técnicos y científicos:

```
1.    Capítulo
  1.1    Sección
    1.1.1  Subsección
```

Es el que ya usa tu plantilla. Confírmalo y no lo mezcles con numeración romana dentro del cuerpo.

### 1.2 La lógica interna de cada capítulo

Regla central del manual, y la que más errores previene: **todos los apartados de un mismo nivel deben estar en el mismo nivel conceptual.** Si 1.1 y 1.2 son secciones hermanas, lo tratado en 1.1.3 debe tener el mismo rango que lo tratado en 1.2.1. Un apartado colocado más arriba o más abajo de donde le corresponde obliga al lector a reordenar mentalmente el texto, y eso es exactamente lo que irrita a un tribunal.

Aplicación a tu Capítulo 1: no cuelgues "Delimitaciones" como 1.6.1 si "Alcance" es 1.6 y ambos tienen el mismo peso; o van los dos al mismo nivel, o uno contiene al otro de forma justificada.

### 1.3 La introducción y el índice se escriben al final

Ambos son producto de lo que terminó siendo la tesis, no de lo que planeabas que fuera. El índice cambia conforme avanzas: capítulos que se parten en dos, secciones que migran, apartados que desaparecen. Esperar a tenerlo todo es lo correcto.

**Pero ojo**: esto habla de la *introducción general* de la memoria, no de tu Capítulo 1 (El Problema), que sí se escribe primero porque es el planteamiento que dirige todo lo demás.

### 1.4 El contrato de las conclusiones — léelo ahora, no al final

Esta es la parte más valiosa del manual para ti, y cambia cómo escribes el Capítulo 1 **hoy**.

Según Soriano, unas conclusiones correctas cumplen:

- **No aportan nada nuevo.** Son el cierre de una investigación terminada; todo hallazgo debe haber aparecido ya en los capítulos previos.
- **Siguen el orden de los capítulos** de los que se extraen, numeradas (primera, segunda, tercera…).
- **Responden a los objetivos o hipótesis** planteados en la introducción.
- **No llevan aparato bibliográfico ni documentación.** Claridad y concisión.
- **Declaran el alcance**: hasta dónde pueden generalizarse los resultados.
- **Sitúan los resultados frente a otras investigaciones**: a cuáles ratifican, con cuáles discrepan.
- **Abren líneas futuras**, pero después de exponer los resultados propios, nunca antes.

**Consecuencia práctica inmediata:** tus cinco objetivos específicos van a necesitar, cada uno, una conclusión que los responda. Si escribes un objetivo que no se puede cerrar con una conclusión verificable, te estás creando un agujero para la defensa. Repasa los cinco con esa prueba antes de dar el Capítulo 1 por cerrado.

El punto de "declarar el alcance" y el de "situar frente a otras investigaciones" encajan perfectamente con tu caso: tus resultados valen solo para Qwen3-8B + A100 + vLLM, y tienes literatura (Sarathi-Serve, DistServe) contra la cual contrastar.

### 1.5 Las cuatro reglas de redacción

Claridad, concisión, naturalidad y exactitud terminológica. En un texto técnico la cuarta es la que más pesa: elige un término para cada concepto y no lo varíes. Si escribes "latencia hasta el primer token" en un párrafo y "TTFT" en el siguiente y "tiempo de respuesta inicial" en el tercero, el lector no sabe si son lo mismo.

Define cada término una sola vez, la primera vez que aparece, y a partir de ahí usa siempre la misma forma.

### 1.6 Errores frecuentes que vas a cometer

Del listado del manual, estos son los que más daño hacen en un texto técnico en español:

- **Queísmo**: cadenas de oraciones subordinadas encadenadas con "que". Produce ambigüedad sobre a qué se refiere cada cosa.
- **Párrafos extensos**: un párrafo, una idea. Si un párrafo tiene cinco oraciones subordinadas, pártelo.
- **Exceso de comas**: sustituye parte de ellas por puntos y puntos y coma.
- **Reiteraciones**: el abuso del mismo término. Aquí hay tensión con la regla de exactitud: los *términos técnicos* no se sustituyen por sinónimos (TTFT es siempre TTFT), pero los verbos y conectores sí.
- **Perífrasis**: "las personas que están integradas en la comunidad" → "las personas integradas en la comunidad".
- **Separar términos que deben ir juntos** por puntuación mal colocada.
- **Conjunciones tras punto y seguido** ("…. Aunque…"): no es correcto con partículas de relación.

### 1.7 Las justas dimensiones

No metas en la memoria todo lo que trabajaste solo porque te costó. Lo que no aporta al argumento va a anexos o se queda fuera. En tu caso esto se traduce en algo concreto: los 320 registros de ejecución no van en el cuerpo del texto; van en el dataset y, si acaso, una muestra representativa en anexos.

### 1.8 Redactar desde el principio

El manual insiste en no dejar la redacción para el final, y en volver al texto tras un receso para corregirlo. Con tu cronograma de 12 semanas esto no es un consejo, es una necesidad: si dejas la memoria para las últimas tres semanas, la entregas sin haberla releído nunca.

---

## 2. Lo que del manual NO debes aplicar

| Recomendación del manual | Por qué no te sirve |
|---|---|
| **Citación clásica a pie de página** (op. cit., ibíd., cfr.) | Tu proyecto usa BibTeX. La UTI casi con seguridad exige APA 7ª edición con citas autor-año en el texto. Mezclar los dos sistemas rompe tu `referencias.bib` y te deja un formato que el lector rechaza. Confirma el estilo con el Ing. Rubio y luego **no lo cambies**. |
| **Capítulos "de autores" vs "de materias"** | Es una taxonomía de tesis de humanidades. Tu memoria sigue la estructura de un trabajo experimental: problema → marco teórico → metodología → resultados → conclusiones. |
| **Capítulos cronológicos vs espaciales** | Misma razón. Tu eje organizador son las fases de inferencia y las variables experimentales, no el tiempo ni el espacio. |
| **Fichas de lectura en papel, cuadernos de campo** | Sustituidos por tu repositorio Git, la bitácora en `docs/bitacora.md` y el gestor bibliográfico. El principio (registrar todo lo que lees, con la referencia completa, en el momento) sigue vigente; el soporte no. |
| **Lista de abreviaturas convencionales** (vid., et alii, pseud.) | Son del uso editorial español en humanidades. Tus abreviaturas son TTFT, TPOT, KV cache, BF16, AWQ, GQA, HBM, NVML, DCGM. Esas sí necesitan un índice de acrónimos al inicio. |

---

## 3. Lo que el manual no cubre y tú sí necesitas

Soriano es de 2008. Tres cosas ausentes que en tu trabajo son centrales:

1. **Cómo se reporta un experimento.** Tu memoria tiene que permitir que otra persona reproduzca tus mediciones: versiones congeladas, semillas, número de repeticiones, orden de ejecución, criterio de descarte. Eso no está en el manual y es el corazón de tu aporte.

2. **Cómo se presentan tablas y figuras.** Cada una numerada, con título, con fuente, citada en el texto antes de aparecer, y legible en escala de grises. Verifica qué exige la plantilla de la UTI.

3. **El uso de IA en la redacción.** No existía la cuestión. Mi recomendación: registra en `docs/bitacora.md` qué partes asististe con IA y cómo verificaste el resultado. Declararlo es defendible; que lo descubran, no. El manual sí dedica un apartado al plagio como riesgo a evitar, y el principio se traslada: responde por cada frase que firmas.

---

## 4. Lista de verificación antes de entregar cualquier capítulo

```
[ ] Todos los apartados de un mismo nivel están en el mismo nivel conceptual
[ ] Numeración decimal consistente, sin mezcla con romanos
[ ] Cada término técnico se define una vez y luego no varía de forma
[ ] Ningún párrafo supera una idea
[ ] Sin queísmo ni cadenas de subordinadas
[ ] Las comas necesarias, ni más ni menos
[ ] Cada tabla y figura: numerada, con título, con fuente, citada en el texto
[ ] Cada cita tiene entrada en referencias.bib y cada entrada está citada
[ ] Estilo de citación único en todo el documento
[ ] Nada en el cuerpo que debería estar en anexos
[ ] El capítulo responde a lo que el índice promete que responde
[ ] Releído al menos una vez tras un día de distancia
```

---

## 5. Prueba de coherencia objetivos ↔ conclusiones

Llena esta tabla ahora, con el Capítulo 1 en la mano. Si alguna fila queda vacía en la columna derecha, ese objetivo está mal planteado.

| Objetivo específico | Conclusión que lo cerrará | ¿Es verificable? |
|---|---|---|
| OE1 Entorno reproducible | | |
| OE2 Benchmark parametrizable | | |
| OE3 Medición de métricas | | |
| OE4 Análisis de fases y cuellos de botella | | |
| OE5 Modelo predictivo | | |

[Observación] OE5 es el que más riesgo tiene: su conclusión depende de alcanzar la meta de MAPE < 15–20 %. Si el modelo no llega, necesitas haber dejado dicho desde el Capítulo 1 que esa meta es **referencial y ajustable tras los experimentos piloto** — que es exactamente lo que dice la ficha. No la conviertas en un compromiso cerrado al redactar.

---

## Referencia del manual consultado

Soriano, R. (2008). *Cómo se escribe una tesis: Guía práctica para estudiantes e investigadores*. Córdoba: Berenice. ISBN 978-84-96756-49-6.

Si lo citas en tu memoria, cítalo así. Pero no lo cites como si fuera una norma institucional: es una guía de oficio, y tu marco normativo es la ficha aprobada más la plantilla de la UTI.
