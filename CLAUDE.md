# CLAUDE.md

## Citación y bibliografía

- **Estilo de citación: IEEE numérico** (confirmado por el tutor). No usar autor-año ni otros estilos.
- Estilo BibTeX: `\bibliographystyle{IEEEtran}` con `\bibliography{referencias}`, definido en `Capítulos/05_Referencias.tex`.
- Base bibliográfica: `referencias.bib`.
- Citar con `\cite{clave}`; para varias fuentes, agruparlas en una sola llamada: `\cite{a,b}` → [1], [2].
- Citar únicamente claves que existan en `referencias.bib`. Si falta una fuente, dejar `% FALTA CITA: <qué haría falta>` en lugar de inventar la clave.
