# Guía para el Capítulo 1 — TFG ITI-TFG-B26-0017

**Modelado experimental de las fases prefill y decode de Qwen3-8B en NVIDIA A100 mediante vLLM**

José Andrés Ureta Granda · Tutor: Andrès Xavier Rubio Proaño
Universidad Tecnológica Indoamérica · Facultad de Ingenierías · ITI · B26 Quito

---

## 1. Archivos de este paquete

| Archivo | Qué contiene | Dónde va |
|---|---|---|
| `informacion-tesis.txt` | Toda la ficha reorganizada: objetivos, alcance, metodología, métricas, recursos, riesgos | Consulta; no se copia tal cual a la memoria |
| `referencias.bib` | 11 referencias verificadas una por una contra su fuente original | Reemplaza o se fusiona con tu `referencias.bib` |
| `guia-capitulo-1.md` | Este documento | Consulta |

---

## 2. Sobre las referencias: qué significa "verificadas"

Cada entrada del `.bib` fue buscada en la web y confirmada contra su fuente primaria (actas del congreso, ACL Anthology, arXiv, ACM DL o el sitio del fabricante) el 16/10/2026. Se comprobó: lista completa de autores, año, nombre exacto del venue, DOI o identificador arXiv y, cuando estaba disponible, el rango de páginas.

**Qué dejé fuera deliberadamente.** La documentación oficial de vLLM, las model cards de Hugging Face y la documentación de NVML/DCGM son fuentes de consulta obligatoria para tu trabajo, pero **no son artículos ni fuentes citables con autoría estable**, y sus URL cambian entre versiones. Van en notas al pie o en un anexo de "entorno de software" con la versión exacta y la fecha de consulta, no en la bibliografía principal.

| Clave BibTeX | Para qué la usas en el Capítulo 1 |
|---|---|
| `vaswani2017attention` | Origen de la arquitectura Transformer |
| `ainslie2023gqa` | GQA: por qué el KV cache de Qwen3-8B es manejable |
| `qwen2025qwen3` | El modelo que evalúas (fuente primaria) |
| `kwon2023pagedattention` | vLLM y PagedAttention: tu motor de inferencia |
| `yu2022orca` | Origen del *continuous batching* |
| `agrawal2024sarathi` | Chunked prefill y el compromiso throughput–latencia |
| `zhong2024distserve` | Prefill compute-bound vs decode memory-bound; TTFT y TPOT |
| `dao2022flashattention` | Algoritmos IO-aware; atención limitada por memoria |
| `lin2024awq` | AWQ: la cuantización que comparas contra BF16 |
| `williams2009roofline` | Marco formal compute-bound vs memory-bound |
| `nvidia2021a100` | Especificaciones de la A100 (ancho de banda, memoria) |

**La cita que más te conviene usar temprano es `zhong2024distserve`.** Afirma explícitamente que *prefill* es compute-bound y *decode* es memory-bound, y distingue TTFT de TPOT como métricas de fases distintas. Es la base bibliográfica directa de tu planteamiento del problema.

---

## 3. Estructura propuesta del Capítulo 1

Antes de escribir, confirma con el Ing. Rubio la estructura que exige la plantilla de la UTI, porque varía entre facultades. La siguiente es la más común en trabajos de integración curricular:

```
CAPÍTULO I — EL PROBLEMA
1.1  Antecedentes
1.2  Planteamiento del problema
1.3  Formulación del problema
1.4  Objetivos
     1.4.1 Objetivo general
     1.4.2 Objetivos específicos
1.5  Justificación
1.6  Alcance y delimitaciones
```

### Contenido mínimo de cada sección

**1.1 Antecedentes.** De lo general a lo particular: crecimiento de los LLM y su costo de despliegue → la inferencia como problema de sistemas, no solo de modelado → trabajos previos de caracterización (Orca, vLLM, Sarathi-Serve, DistServe) → qué queda sin resolver. Aquí es donde van las citas.

**1.2 Planteamiento del problema.** Copia el argumento central de la ficha: las métricas agregadas (tokens/s) ocultan el comportamiento diferenciado de prefill y decode, y por eso no explican por qué una configuración mejora el throughput y empeora la latencia interactiva. Las cinco preguntas sin respuesta de la ficha son el corazón de esta sección.

**1.3 Formulación.** Una sola pregunta de investigación, redactada como pregunta. Algo en la línea de: *¿Cómo afectan la longitud de contexto, la concurrencia y la cuantización al comportamiento de las fases prefill y decode de Qwen3-8B sobre una GPU NVIDIA A100 con vLLM?*

**1.4 Objetivos.** Van literales de la ficha. No los reformules: deben coincidir con el acta aprobada.

**1.5 Justificación.** Tres ángulos: técnico (configurar eficientemente un LLM en infraestructura GPU limitada), institucional (primer caso de estudio del proyecto LLM-CoDesign) y social (universidades y laboratorios sin acceso a infraestructura hiperescalable).

**1.6 Alcance y delimitaciones.** La tabla de variables experimentales y la lista de exclusiones de la ficha. Esta sección te protege en la defensa: todo lo que declares fuera de alcance no te lo pueden reclamar después.

---

## 4. Tres errores que arruinan un Capítulo 1

1. **Escribir el marco teórico aquí.** Explicar qué es un Transformer, qué es PagedAttention o cómo funciona AWQ pertenece al Capítulo 2. En el Capítulo 1 solo se mencionan al pasar, con cita.

2. **Objetivos que no coinciden con el acta.** Si "reformulas para que suene mejor", generas una discrepancia con el documento institucional aprobado. Los cinco objetivos específicos van tal como están.

3. **Anticipar resultados.** No escribas "se espera demostrar que la cuantización reduce la latencia". Aún no has medido nada. El Capítulo 1 plantea el problema; no adelanta conclusiones.

---

## 5. Prompt para Claude Code

Copia el bloque completo. Está dividido en fases a propósito: en el primer paso el agente solo diagnostica, y tú apruebas antes de que escriba.

````
Contexto: proyecto LaTeX de la memoria del TFG ITI-TFG-B26-0017 en este
repositorio. main.tex compila con pdfLaTeX (NUNCA con XeLaTeX: la ruta
XeLaTeX exige la fuente Century Gothic, que no está instalada y produce
PDF en blanco). Existen las carpetas Capitulos/, Preliminares/, Anexos/
y el archivo referencias.bib.

Datos del trabajo:
- Título: Modelado experimental de las fases prefill y decode de Qwen3-8B
  en NVIDIA A100 mediante vLLM
- Autor: José Andrés Ureta Granda
- Tutor: Andrès Xavier Rubio Proaño
- Universidad Tecnológica Indoamérica, Facultad de Ingenierías,
  Ingeniería en Tecnologías de la Información

Te adjunto dos archivos: informacion-tesis.txt (toda la información de la
ficha institucional aprobada) y referencias.bib (referencias ya
verificadas contra sus fuentes originales).

=== FASE 1 — DIAGNÓSTICO (no escribas ni modifiques nada) ===

1. Lee informacion-tesis.txt completo.
2. Localiza el archivo .tex del Capítulo 1 dentro de Capitulos/ y
   muéstramelo. Dime qué estructura de secciones ya trae la plantilla
   y qué comandos usa para secciones, citas y referencias cruzadas.
3. Revisa main.tex y dime: qué paquete de bibliografía usa
   (natbib, biblatex, bibtex plano), qué estilo de citación está
   configurado, y cómo se incluyen los capítulos.
4. Revisa referencias.bib y confirma que las 11 entradas están
   bien formadas.
5. Propón un esquema de secciones para el Capítulo 1 respetando la
   estructura que ya traiga la plantilla.

Detente aquí y espera mi aprobación del esquema.

=== FASE 2 — REDACCIÓN (solo tras mi visto bueno) ===

Escribe el Capítulo 1 en el archivo .tex correspondiente, con estas
secciones:

1.1 Antecedentes
1.2 Planteamiento del problema
1.3 Formulación del problema
1.4 Objetivos (general y los cinco específicos)
1.5 Justificación
1.6 Alcance y delimitaciones

REGLAS ESTRICTAS:

- Los objetivos van LITERALES tal como aparecen en informacion-tesis.txt.
  No los reformules ni los "mejores": deben coincidir con el acta
  institucional aprobada.
- Cita ÚNICAMENTE claves que existan en referencias.bib. Si necesitas
  respaldar una afirmación y no hay una clave adecuada, marca el punto
  con un comentario LaTeX "% FALTA CITA: <qué haría falta>" y
  repórtamelo al final. NO inventes referencias, autores, años, DOIs
  ni títulos bajo ninguna circunstancia.
- No escribas marco teórico. No expliques qué es un Transformer, qué
  es PagedAttention, qué es AWQ ni cómo funciona el KV cache: eso
  pertenece al Capítulo 2. Aquí solo se mencionan con cita.
- No anticipes resultados ni conclusiones. Nada de "se espera
  demostrar que...". Aún no se ha medido nada.
- No inventes cifras. Las únicas cifras permitidas son las que están
  en informacion-tesis.txt (variables experimentales, número de
  configuraciones, duración, metas de MAPE). Las cifras del ejemplo
  ilustrativo de la ficha (TTFT 340 ms, TPOT 24 ms/token, etc.) NO
  son resultados y no deben aparecer.
- Español académico formal, tercera persona impersonal ("se analiza",
  "se evalúa"). Nunca primera persona.
- Extensión: entre 1800 y 2500 palabras.
- Usa \label{} en cada sección con el prefijo sec:cap1:
- No modifiques uti_tfm.sty, main.tex, márgenes, fuentes ni
  interlineado: es formato institucional validado por la universidad.
- No toques ningún archivo fuera del .tex del Capítulo 1.

=== FASE 3 — VERIFICACIÓN ===

1. Muéstrame el diff exacto de lo que escribiste.
2. Lista todas las claves de cita que usaste y confirma que cada una
   existe en referencias.bib.
3. Lista todos los comentarios "% FALTA CITA" que dejaste.
4. Compila con: pdflatex main.tex, bibtex main, pdflatex main.tex,
   pdflatex main.tex. Repórtame errores y advertencias.
5. Dime si quedaron citas sin resolver (buscando "Citation ... undefined"
   en main.log).

Antes de compilar, avísame para que cierre main.pdf en el visor
(Windows bloquea el archivo).
````

---

## 6. Después de generar el borrador

El agente produce un borrador, no un capítulo entregable. Revisa en este orden:

1. **Las citas.** Abre el PDF y verifica que cada cita tenga sentido donde está. Un agente puede citar `williams2009roofline` en un párrafo donde no corresponde.
2. **Los objetivos.** Compáralos palabra por palabra con la ficha.
3. **Los `% FALTA CITA`.** Cada uno es una afirmación sin respaldo. O consigues la fuente (y la verificas) o eliminas la afirmación.
4. **El tono.** Si suena a texto generado, reescribe esos párrafos tú. Tu tutor y dos lectores van a leer esto.

Y registra en `docs/bitacora.md` la fecha y la decisión: *"Capítulo 1 redactado con asistencia de IA a partir de la ficha institucional; referencias verificadas manualmente contra fuentes primarias."* La transparencia sobre el uso de IA es defendible; ocultarla no.
