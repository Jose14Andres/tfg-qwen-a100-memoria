from datetime import datetime, timezone

import pytest

from tfgbench.entorno import capturar_entorno, formato_gpu, picos
from tfgbench.ids import hash_contenido, nuevo_uuid, semilla_derivada
from tfgbench.schemas import CLAVES_PRIMARIAS, TABLAS, ErrorEsquema, a_tabla


def test_hash_independiente_del_orden_de_claves():
    assert hash_contenido({"a": 1, "b": 2}) == hash_contenido({"b": 2, "a": 1})
    assert len(hash_contenido({"a": 1})) == 16


def test_uuid_y_semilla():
    assert nuevo_uuid() != nuevo_uuid()
    assert semilla_derivada("e1", 128) == semilla_derivada("e1", 128)
    assert 0 <= semilla_derivada("x") < 2**31


def test_claves_primarias_existen_en_esquemas():
    assert set(CLAVES_PRIMARIAS) == set(TABLAS)
    for tabla, claves in CLAVES_PRIMARIAS.items():
        for c in claves:
            assert c in TABLAS[tabla].names
            assert not TABLAS[tabla].field(c).nullable


def _peticion(**cambios):
    fila = {"req_id": "r1", "run_id": "c1", "indice": 0, "es_warmup": False, "tokens_prompt": 128,
            "tokens_salida": 16, "t_envio_ns": 0, "t_primer_token_ns": 10, "t_fin_ns": 20,
            "ttft_ms": None, "tpot_ms": None, "e2e_ms": None, "estado": "ok", "codigo_error": None}
    fila.update(cambios)
    return fila


def test_a_tabla_valida():
    t = a_tabla("peticion", [_peticion()])
    assert t.num_rows == 1
    assert t.schema == TABLAS["peticion"]


def test_a_tabla_rechaza_enum_invalido():
    with pytest.raises(ErrorEsquema, match="estado"):
        a_tabla("peticion", [_peticion(estado="raro")])


def test_a_tabla_rechaza_nulo_no_permitido():
    with pytest.raises(ErrorEsquema):
        a_tabla("peticion", [_peticion(tokens_prompt=None)])


def test_a_tabla_rechaza_columnas_extra():
    with pytest.raises(ErrorEsquema, match="extra"):
        a_tabla("peticion", [_peticion(otra=1)])


def test_formato_y_picos():
    assert formato_gpu("NVIDIA A100-SXM4-80GB") == "sxm4"
    assert formato_gpu("NVIDIA A100 80GB PCIe") == "pcie"
    assert picos("sxm4", 79.6) == (312.0, 2039.0)
    assert picos("pcie", 40.0)[1] == 1555.0


def test_entorno_sin_gpu_cumple_esquema():
    fila = capturar_entorno("pruebas", permitir_sin_gpu=True)
    assert isinstance(fila["registrado_en"], datetime) and fila["registrado_en"].tzinfo == timezone.utc
    a_tabla("entorno", [fila])
