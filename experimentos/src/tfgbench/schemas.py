"""Esquemas Arrow/Parquet de cada tabla (Modelo de Datos §5).

Esta es la única fuente de verdad del modelo físico: los escritores (runner) y
los lectores (análisis) validan contra estos esquemas.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pyarrow as pa

TEXT = pa.string()
INT32 = pa.int32()
INT64 = pa.int64()
REAL = pa.float64()
BOOL = pa.bool_()
TIMESTAMP = pa.timestamp("us", tz="UTC")


def _f(nombre: str, tipo: pa.DataType, nulo: bool = False) -> pa.Field:
    return pa.field(nombre, tipo, nullable=nulo)


ENTORNO = pa.schema([
    _f("env_id", TEXT),
    _f("gpu_modelo", TEXT),
    _f("gpu_uuid", TEXT),
    _f("gpu_memoria_gb", REAL),
    _f("gpu_formato", TEXT),
    _f("ancho_banda_pico_gbs", REAL),
    _f("flops_pico_tflops", REAL),
    _f("driver_version", TEXT),
    _f("cuda_version", TEXT),
    _f("cpu_modelo", TEXT, True),
    _f("ram_gb", REAL, True),
    _f("so", TEXT),
    _f("python_version", TEXT),
    _f("vllm_version", TEXT),
    _f("torch_version", TEXT),
    _f("imagen_contenedor", TEXT, True),
    _f("lockfile_sha256", TEXT),
    _f("proveedor", TEXT),
    _f("registrado_en", TIMESTAMP),
])

EXPERIMENTO = pa.schema([
    _f("exp_id", TEXT),
    _f("nombre", TEXT),
    _f("fase_objetivo", TEXT),
    _f("modo_carga", TEXT),
    _f("descripcion", TEXT, True),
])

CONFIG_SERVIDOR = pa.schema([
    _f("config_id", TEXT),
    _f("modelo_id", TEXT),
    _f("modelo_revision", TEXT),
    _f("dtype", TEXT),
    _f("max_model_len", INT32),
    _f("gpu_memory_utilization", REAL),
    _f("chunked_prefill", BOOL),
    _f("max_num_batched_tokens", INT32, True),
    _f("max_num_seqs", INT32, True),
    _f("prefix_caching", BOOL),
    _f("enforce_eager", BOOL),
    _f("temperatura", REAL),
    _f("ignore_eos", BOOL),
    _f("semilla", INT32),
    _f("args_cli", TEXT),
])

CORRIDA = pa.schema([
    _f("run_id", TEXT),
    _f("exp_id", TEXT),
    _f("env_id", TEXT),
    _f("config_id", TEXT),
    _f("l_in", INT32),
    _f("l_out", INT32),
    _f("concurrencia", INT32),
    _f("tasa_llegada", REAL, True),
    _f("num_peticiones", INT32),
    _f("num_warmup", INT32),
    _f("repeticion", INT32),
    _f("semilla_prompts", INT32),
    _f("kv_capacidad_tokens", INT64, True),
    _f("inicio", TIMESTAMP),
    _f("fin", TIMESTAMP, True),
    _f("estado", TEXT),
    _f("notas", TEXT, True),
])

PETICION = pa.schema([
    _f("req_id", TEXT),
    _f("run_id", TEXT),
    _f("indice", INT32),
    _f("es_warmup", BOOL),
    _f("tokens_prompt", INT32),
    _f("tokens_salida", INT32, True),
    _f("t_envio_ns", INT64),
    _f("t_primer_token_ns", INT64, True),
    _f("t_fin_ns", INT64, True),
    _f("ttft_ms", REAL, True),
    _f("tpot_ms", REAL, True),
    _f("e2e_ms", REAL, True),
    _f("estado", TEXT),
    _f("codigo_error", TEXT, True),
])

EVENTO_TOKEN = pa.schema([
    _f("req_id", TEXT),
    _f("idx_token", INT32),
    _f("t_ns", INT64),
    _f("n_tokens_chunk", INT32),
    _f("itl_ms", REAL, True),
])

MUESTRA_GPU = pa.schema([
    _f("run_id", TEXT),
    _f("t_ns", INT64),
    _f("util_gpu_pct", REAL),
    _f("util_mem_pct", REAL),
    _f("mem_usada_mib", REAL),
    _f("potencia_w", REAL),
    _f("reloj_sm_mhz", INT32),
    _f("reloj_mem_mhz", INT32),
    _f("temperatura_c", REAL),
    _f("throttle_razones", TEXT, True),
])

MUESTRA_VLLM = pa.schema([
    _f("run_id", TEXT),
    _f("t_ns", INT64),
    _f("pet_ejecutando", INT32),
    _f("pet_esperando", INT32),
    _f("kv_cache_uso_pct", REAL),
    _f("preemptions_total", INT64),
    _f("tokens_prompt_total", INT64),
    _f("tokens_generados_total", INT64),
])

AJUSTE_MODELO = pa.schema([
    _f("fit_id", TEXT),
    _f("fase", TEXT),
    _f("env_id", TEXT),
    _f("config_id", TEXT),
    _f("formula", TEXT),
    _f("metodo", TEXT),
    _f("variable_respuesta", TEXT),
    _f("semilla_particion", INT32),
    _f("frac_entrenamiento", REAL),
    _f("n_entrenamiento", INT32),
    _f("n_prueba", INT32),
    _f("mape_pct", REAL),
    _f("r2", REAL),
    _f("rmse_ms", REAL),
    _f("rango_validez", TEXT),
    _f("creado_en", TIMESTAMP),
])

COEFICIENTE = pa.schema([
    _f("fit_id", TEXT),
    _f("nombre", TEXT),
    _f("valor", REAL),
    _f("error_estandar", REAL, True),
    _f("ic95_inf", REAL, True),
    _f("ic95_sup", REAL, True),
    _f("unidad", TEXT),
])

AJUSTE_CORRIDA = pa.schema([
    _f("fit_id", TEXT),
    _f("run_id", TEXT),
    _f("particion", TEXT),
])

TABLAS: dict[str, pa.Schema] = {
    "entorno": ENTORNO,
    "experimento": EXPERIMENTO,
    "config_servidor": CONFIG_SERVIDOR,
    "corrida": CORRIDA,
    "peticion": PETICION,
    "evento_token": EVENTO_TOKEN,
    "muestra_gpu": MUESTRA_GPU,
    "muestra_vllm": MUESTRA_VLLM,
    "ajuste_modelo": AJUSTE_MODELO,
    "coeficiente": COEFICIENTE,
    "ajuste_corrida": AJUSTE_CORRIDA,
}

CLAVES_PRIMARIAS: dict[str, tuple[str, ...]] = {
    "entorno": ("env_id",),
    "experimento": ("exp_id",),
    "config_servidor": ("config_id",),
    "corrida": ("run_id",),
    "peticion": ("req_id",),
    "evento_token": ("req_id", "idx_token"),
    "muestra_gpu": ("run_id", "t_ns"),
    "muestra_vllm": ("run_id", "t_ns"),
    "ajuste_modelo": ("fit_id",),
    "coeficiente": ("fit_id", "nombre"),
    "ajuste_corrida": ("fit_id", "run_id"),
}

# Enumeraciones cerradas: (tabla, columna) -> valores permitidos
ENUMS: dict[tuple[str, str], frozenset[str]] = {
    ("entorno", "gpu_formato"): frozenset({"pcie", "sxm4", "desconocido"}),
    ("experimento", "exp_id"): frozenset({"e1", "e2", "e3", "e4", "e5", "piloto"}),
    ("experimento", "fase_objetivo"): frozenset({"prefill", "decode", "e2e", "interferencia", "capacidad"}),
    ("experimento", "modo_carga"): frozenset({"offline", "online"}),
    ("config_servidor", "dtype"): frozenset({"bfloat16", "float16"}),
    ("corrida", "estado"): frozenset({"ok", "oom", "timeout", "error", "abortada"}),
    ("peticion", "estado"): frozenset({"ok", "error", "abortada", "timeout"}),
    ("ajuste_modelo", "fase"): frozenset({"prefill", "decode", "e2e", "throughput"}),
    ("ajuste_modelo", "metodo"): frozenset({"ols", "nnls", "wls", "por_tramos"}),
    ("ajuste_corrida", "particion"): frozenset({"entrenamiento", "prueba"}),
}


class ErrorEsquema(ValueError):
    pass


def validar_enums(tabla: str, filas: Iterable[Mapping[str, Any]]) -> None:
    for fila in filas:
        for (t, col), permitidos in ENUMS.items():
            if t == tabla and col in fila and fila[col] is not None and fila[col] not in permitidos:
                raise ErrorEsquema(f"{tabla}.{col}={fila[col]!r} no está en {sorted(permitidos)}")


def a_tabla(tabla: str, filas: list[Mapping[str, Any]]) -> pa.Table:
    """Convierte filas (dicts) en una tabla Arrow validada contra el esquema.

    Rechaza columnas desconocidas, faltantes, nulos no permitidos y valores
    fuera de las enumeraciones.
    """
    esquema = TABLAS[tabla]
    nombres = set(esquema.names)
    for i, fila in enumerate(filas):
        extra = set(fila) - nombres
        falta = nombres - set(fila)
        if extra or falta:
            raise ErrorEsquema(f"{tabla}[{i}]: columnas extra={sorted(extra)} faltantes={sorted(falta)}")
    validar_enums(tabla, filas)
    columnas = {n: [f[n] for f in filas] for n in esquema.names}
    try:
        tabla_arrow = pa.Table.from_pydict(columnas, schema=esquema)
        tabla_arrow.validate(full=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
        raise ErrorEsquema(f"{tabla}: {e}") from e
    for campo in esquema:
        if not campo.nullable and tabla_arrow.column(campo.name).null_count:
            raise ErrorEsquema(f"{tabla}.{campo.name} no admite nulos")
    return tabla_arrow
