"""Identificadores del modelo de datos (Modelo de Datos §2).

- Entidades de contexto (entorno, config_servidor): hash del contenido canónico,
  de modo que dos configuraciones idénticas producen el mismo identificador.
- Entidades de ejecución (corrida, peticion, ajuste_modelo): UUID v4.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

LONGITUD_HASH = 16


def json_canonico(datos: Any) -> str:
    """Serializa a JSON con claves ordenadas y sin espacios (forma canónica)."""
    return json.dumps(datos, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_contenido(datos: Any) -> str:
    """SHA-256 truncado a 16 caracteres hexadecimales del JSON canónico."""
    return hashlib.sha256(json_canonico(datos).encode("utf-8")).hexdigest()[:LONGITUD_HASH]


def nuevo_uuid() -> str:
    return str(uuid.uuid4())


def semilla_derivada(*partes: Any) -> int:
    """Semilla de 31 bits determinista a partir de cualquier combinación de valores."""
    return int(hashlib.sha256(json_canonico(list(partes)).encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
