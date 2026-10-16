"""Prompts sintéticos de longitud exacta (SPEC RF2).

Se generan directamente como listas de IDs de token y se envían a vLLM como
`prompt_token_ids` (API offline) o como lista de enteros en `prompt`
(endpoint /v1/completions). Así la longitud es exacta por construcción, sin
depender de que decodificar y volver a tokenizar texto conserve el conteo.

El contenido no afecta al costo de cómputo de prefill ni de decode (las
operaciones son las mismas para cualquier token); por eso basta con IDs
pseudoaleatorios deterministas. Solo se excluyen los tokens especiales.
"""

from __future__ import annotations

import numpy as np

# Qwen3 usa el vocabulario de Qwen2: IDs 0..151642 son tokens regulares y los
# tokens especiales empiezan en 151643 (<|endoftext|>).
# VERIFICAR contra tokenizer_config.json de la revisión fijada (ver ESTADO.md).
VOCAB_REGULAR = 151_643


def prompt_ids(l_in: int, semilla: int, indice: int, vocab: int = VOCAB_REGULAR) -> list[int]:
    """Prompt determinista de exactamente `l_in` tokens para la petición `indice`."""
    if l_in <= 0:
        raise ValueError("l_in debe ser positivo")
    rng = np.random.default_rng(np.random.SeedSequence([semilla, indice]))
    return rng.integers(0, vocab, size=l_in, dtype=np.int64).tolist()


def lote_prompts(l_in: int, n: int, semilla: int, desde: int = 0, vocab: int = VOCAB_REGULAR) -> list[list[int]]:
    """`n` prompts distintos (índices desde..desde+n-1) de longitud `l_in`."""
    return [prompt_ids(l_in, semilla, i, vocab) for i in range(desde, desde + n)]
