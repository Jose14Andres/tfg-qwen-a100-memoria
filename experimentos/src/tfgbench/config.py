"""Carga de configuración y expansión de la matriz experimental (SPEC §7–8).

Un archivo de experimento declara metadatos, ajustes del servidor (sobre la
base común) y uno o más *barridos*. Cada barrido es un producto cartesiano de
factores; la matriz del experimento es la unión de sus barridos.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .ids import hash_contenido, semilla_derivada
from .schemas import ENUMS


class ErrorConfig(ValueError):
    pass


@dataclass(frozen=True)
class ServerConfig:
    """Parámetros de vLLM. Cualquier cambio produce un config_id distinto."""

    modelo_id: str = "Qwen/Qwen3-8B"
    modelo_revision: str = "main"  # fijar el hash del commit de Hugging Face antes de la campaña
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    chunked_prefill: bool = True
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = 256
    prefix_caching: bool = False
    enforce_eager: bool = False
    temperatura: float = 0.0
    ignore_eos: bool = True
    semilla: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in ENUMS[("config_servidor", "dtype")]:
            raise ErrorConfig(f"dtype no soportado: {self.dtype}")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ErrorConfig("gpu_memory_utilization debe estar en (0, 1]")
        if self.prefix_caching:
            # Regla R8 del modelo de datos.
            raise ErrorConfig("prefix_caching debe ser false en toda la campaña (regla R8)")

    @property
    def config_id(self) -> str:
        return hash_contenido(asdict(self))

    def args_cli(self) -> list[str]:
        """Argumentos para `vllm serve`.

        Las banderas dependen de la versión de vLLM: verificar con
        `vllm serve --help` de la versión fijada y ajustar solo aquí.
        """
        args = [
            self.modelo_id,
            "--revision", self.modelo_revision,
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--enable-chunked-prefill" if self.chunked_prefill else "--no-enable-chunked-prefill",
            "--enable-prefix-caching" if self.prefix_caching else "--no-enable-prefix-caching",
            "--seed", str(self.semilla),
        ]
        if self.max_num_batched_tokens is not None:
            args += ["--max-num-batched-tokens", str(self.max_num_batched_tokens)]
        if self.max_num_seqs is not None:
            args += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.enforce_eager:
            args.append("--enforce-eager")
        return args


@dataclass(frozen=True)
class RunSpec:
    """Un punto de la matriz en una repetición: lo que luego será una `corrida`."""

    exp_id: str
    servidor: ServerConfig
    l_in: int
    l_out: int
    concurrencia: int
    tasa_llegada: float | None
    repeticion: int
    num_peticiones: int
    num_warmup: int

    @property
    def semilla_prompts(self) -> int:
        # No depende de la configuración del servidor: las corridas con chunked
        # prefill on/off usan exactamente los mismos prompts (comparación pareada).
        return semilla_derivada(self.exp_id, self.l_in, self.l_out, self.concurrencia,
                                self.tasa_llegada, self.repeticion)

    @property
    def tokens_en_vuelo(self) -> int:
        return self.concurrencia * (self.l_in + self.l_out)


@dataclass(frozen=True)
class Experimento:
    exp_id: str
    nombre: str
    fase_objetivo: str
    modo_carga: str
    descripcion: str
    servidor: ServerConfig
    barridos: tuple[dict[str, Any], ...]
    repeticiones: int
    num_warmup: int
    limite_tokens_kv: int
    lotes_por_corrida: int | None = None
    num_peticiones: int | None = None
    descartes: list[dict[str, Any]] = field(default_factory=list)

    def fila_experimento(self) -> dict[str, Any]:
        return {"exp_id": self.exp_id, "nombre": self.nombre, "fase_objetivo": self.fase_objetivo,
                "modo_carga": self.modo_carga, "descripcion": self.descripcion}

    def expandir(self) -> list[RunSpec]:
        """Matriz completa (sin duplicados), descartando puntos inviables.

        Descarta y registra en `self.descartes`:
        - l_in + l_out > max_model_len
        - concurrencia * (l_in + l_out) > limite_tokens_kv (en offline, el lote
          dejaría de ejecutarse como un único lote y la medición perdería sentido)
        """
        self.descartes.clear()
        vistos: set[tuple] = set()
        specs: list[RunSpec] = []
        for barrido in self.barridos:
            ejes = {
                "l_in": barrido["l_in"],
                "l_out": barrido["l_out"],
                "concurrencia": barrido["concurrencia"],
                "tasa_llegada": barrido.get("tasa_llegada", [None]),
                "chunked_prefill": barrido.get("chunked_prefill", [self.servidor.chunked_prefill]),
            }
            for l_in, l_out, b, tasa, cp in itertools.product(*ejes.values()):
                punto = {"l_in": l_in, "l_out": l_out, "concurrencia": b, "tasa_llegada": tasa,
                         "chunked_prefill": cp}
                if l_in + l_out > self.servidor.max_model_len:
                    self.descartes.append({**punto, "motivo": "excede max_model_len"})
                    continue
                if self.modo_carga == "offline" and b * (l_in + l_out) > self.limite_tokens_kv:
                    self.descartes.append({**punto, "motivo": "excede limite_tokens_kv"})
                    continue
                servidor = replace(self.servidor, chunked_prefill=cp)
                for rep in range(1, self.repeticiones + 1):
                    clave = (servidor.config_id, l_in, l_out, b, tasa, rep)
                    if clave in vistos:
                        continue
                    vistos.add(clave)
                    n = self.num_peticiones if self.num_peticiones is not None else self.lotes_por_corrida * b
                    specs.append(RunSpec(self.exp_id, servidor, l_in, l_out, b, tasa, rep, n, self.num_warmup))
        return specs


def _leer_yaml(ruta: Path) -> dict[str, Any]:
    with open(ruta, encoding="utf-8") as f:
        datos = yaml.safe_load(f) or {}
    if not isinstance(datos, dict):
        raise ErrorConfig(f"{ruta}: se esperaba un mapeo YAML")
    return datos


def _servidor(base: dict[str, Any], ajustes: dict[str, Any]) -> ServerConfig:
    campos = {**base, **ajustes}
    desconocidos = set(campos) - set(ServerConfig.__dataclass_fields__)
    if desconocidos:
        raise ErrorConfig(f"campos de servidor desconocidos: {sorted(desconocidos)}")
    return ServerConfig(**campos)


def cargar_experimento(ruta: str | Path, ruta_base: str | Path | None = None) -> Experimento:
    ruta = Path(ruta)
    datos = _leer_yaml(ruta)
    ruta_base = Path(ruta_base) if ruta_base else ruta.parent / "servidor_base.yaml"
    base = _leer_yaml(ruta_base) if ruta_base.exists() else {}
    limite = base.pop("limite_tokens_kv", None)

    exp = datos.get("experimento") or {}
    for clave in ("exp_id", "nombre", "fase_objetivo", "modo_carga"):
        if clave not in exp:
            raise ErrorConfig(f"{ruta}: falta experimento.{clave}")
    for col in ("exp_id", "fase_objetivo", "modo_carga"):
        if exp[col] not in ENUMS[("experimento", col)]:
            raise ErrorConfig(f"{ruta}: experimento.{col}={exp[col]!r} no válido")

    ejecucion = datos.get("ejecucion") or {}
    lotes = ejecucion.get("lotes_por_corrida")
    num_pet = ejecucion.get("num_peticiones")
    if (lotes is None) == (num_pet is None):
        raise ErrorConfig(f"{ruta}: indicar exactamente uno de ejecucion.lotes_por_corrida o ejecucion.num_peticiones")

    barridos = datos.get("barridos")
    if not barridos:
        raise ErrorConfig(f"{ruta}: se requiere al menos un barrido")
    for i, b in enumerate(barridos):
        for eje in ("l_in", "l_out", "concurrencia"):
            if not b.get(eje):
                raise ErrorConfig(f"{ruta}: barrido {i} sin niveles para {eje}")
        if exp["modo_carga"] == "online" and not b.get("tasa_llegada"):
            raise ErrorConfig(f"{ruta}: barrido {i} online requiere tasa_llegada")

    return Experimento(
        exp_id=exp["exp_id"],
        nombre=exp["nombre"],
        fase_objetivo=exp["fase_objetivo"],
        modo_carga=exp["modo_carga"],
        descripcion=exp.get("descripcion", ""),
        servidor=_servidor(base, datos.get("servidor") or {}),
        barridos=tuple(barridos),
        repeticiones=int(ejecucion.get("repeticiones", 5)),
        num_warmup=int(ejecucion.get("num_warmup", 5)),
        limite_tokens_kv=int(ejecucion.get("limite_tokens_kv", limite or 300_000)),
        lotes_por_corrida=lotes,
        num_peticiones=num_pet,
    )
