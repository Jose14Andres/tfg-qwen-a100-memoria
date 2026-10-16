"""Captura del entorno de ejecución (tabla `entorno`)."""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .ids import hash_contenido

# Picos de la hoja de datos de NVIDIA A100 (BF16 Tensor Core denso, HBM).
# VERIFICAR contra la variante exacta usada en la campaña.
PICOS_A100: dict[tuple[str, int], tuple[float, float]] = {
    # (formato, memoria GB aprox.) -> (TFLOPS BF16, GB/s)
    ("sxm4", 80): (312.0, 2039.0),
    ("sxm4", 40): (312.0, 1555.0),
    ("pcie", 80): (312.0, 1935.0),
    ("pcie", 40): (312.0, 1555.0),
}


def _version(paquete: str) -> str:
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return "no-instalado"


def _nvidia_smi() -> dict[str, str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    campos = "name,uuid,memory.total,driver_version"
    salida = subprocess.run(
        ["nvidia-smi", f"--query-gpu={campos}", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip()
    nombre, uuid, mem_mib, driver = [x.strip() for x in salida.split(",")]
    cabecera = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    cuda = "desconocido"
    for parte in cabecera.split("|"):
        if "CUDA Version" in parte:
            cuda = parte.split("CUDA Version:")[1].strip().split()[0]
    return {"name": nombre, "uuid": uuid, "mem_mib": mem_mib, "driver": driver, "cuda": cuda}


def formato_gpu(nombre: str) -> str:
    n = nombre.upper()
    if "SXM" in n:
        return "sxm4"
    if "PCIE" in n:
        return "pcie"
    return "desconocido"


def picos(formato: str, memoria_gb: float) -> tuple[float, float]:
    clave = (formato, 80 if memoria_gb > 60 else 40)
    if clave not in PICOS_A100:
        raise KeyError(f"Sin picos registrados para {clave}; añadirlos a PICOS_A100")
    return PICOS_A100[clave]


def _ram_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for linea in meminfo.read_text().splitlines():
        if linea.startswith("MemTotal:"):
            return round(int(linea.split()[1]) / 1024**2, 1)
    return None


def _hash_archivo(ruta: Path | None) -> str:
    if ruta is None or not ruta.exists():
        return "sin-lockfile"
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def capturar_entorno(proveedor: str, lockfile: Path | None = None, imagen_contenedor: str | None = None,
                     permitir_sin_gpu: bool = False) -> dict[str, Any]:
    """Fila de la tabla `entorno` para la máquina actual.

    `permitir_sin_gpu` solo sirve para desarrollo y pruebas en una máquina sin
    GPU NVIDIA; esas filas nunca deben mezclarse con datos de la campaña.
    """
    gpu = _nvidia_smi()
    if gpu is None:
        if not permitir_sin_gpu:
            raise RuntimeError("nvidia-smi no disponible: no se puede registrar el entorno de la campaña")
        gpu = {"name": "SIN-GPU", "uuid": "sin-gpu", "mem_mib": "0", "driver": "n/a", "cuda": "n/a"}
    memoria_gb = float(gpu["mem_mib"]) / 1024
    formato = formato_gpu(gpu["name"])
    try:
        tflops, gbs = picos(formato, memoria_gb)
    except KeyError:
        if not permitir_sin_gpu:
            raise
        tflops, gbs = 0.0, 0.0

    fila: dict[str, Any] = {
        "gpu_modelo": gpu["name"],
        "gpu_uuid": gpu["uuid"],
        "gpu_memoria_gb": round(memoria_gb, 2),
        "gpu_formato": formato,
        "ancho_banda_pico_gbs": gbs,
        "flops_pico_tflops": tflops,
        "driver_version": gpu["driver"],
        "cuda_version": gpu["cuda"],
        "cpu_modelo": platform.processor() or None,
        "ram_gb": _ram_gb(),
        "so": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.split()[0],
        "vllm_version": _version("vllm"),
        "torch_version": _version("torch"),
        "imagen_contenedor": imagen_contenedor,
        "lockfile_sha256": _hash_archivo(lockfile),
        "proveedor": proveedor,
    }
    fila["env_id"] = hash_contenido(fila)
    fila["registrado_en"] = datetime.now(timezone.utc)
    return fila
