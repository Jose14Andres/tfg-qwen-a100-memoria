from pathlib import Path

import pytest

from tfgbench.config import ErrorConfig, ServerConfig, cargar_experimento

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _yaml(tmp_path: Path, texto: str) -> Path:
    (tmp_path / "servidor_base.yaml").write_text("max_model_len: 4096\nlimite_tokens_kv: 10000\n", encoding="utf-8")
    ruta = tmp_path / "exp.yaml"
    ruta.write_text(texto, encoding="utf-8")
    return ruta


BASICO = """
experimento: {exp_id: e1, nombre: t, fase_objetivo: prefill, modo_carga: offline}
ejecucion: {repeticiones: 2, num_warmup: 1, lotes_por_corrida: 3}
barridos:
  - l_in: [100, 1000, 4000]
    l_out: [1]
    concurrencia: [1, 8]
    chunked_prefill: [false, true]
"""


def test_config_id_estable_y_sensible_a_cambios():
    a, b = ServerConfig(), ServerConfig()
    assert a.config_id == b.config_id
    assert ServerConfig(chunked_prefill=False).config_id != a.config_id


def test_prefix_caching_prohibido():
    with pytest.raises(ErrorConfig):
        ServerConfig(prefix_caching=True)


def test_args_cli_reflejan_config():
    args = ServerConfig(chunked_prefill=False, enforce_eager=True).args_cli()
    assert args[0] == "Qwen/Qwen3-8B"
    assert "--no-enable-chunked-prefill" in args
    assert "--no-enable-prefix-caching" in args
    assert "--enforce-eager" in args


def test_expansion_y_descartes(tmp_path):
    exp = cargar_experimento(_yaml(tmp_path, BASICO))
    specs = exp.expandir()
    # 3 l_in x 2 B x 2 CP x 2 rep = 24, menos descartes:
    # l_in=4000,l_out=1 cabe en 4096; B=8 → 8*4001 > 10000 y 8*1001 = 8008 ≤ 10000
    descartados = {(d["l_in"], d["concurrencia"]) for d in exp.descartes}
    assert descartados == {(4000, 8)}
    assert len(specs) == 24 - 1 * 2 * 2
    assert all(s.num_peticiones == 3 * s.concurrencia for s in specs)
    assert all(s.tokens_en_vuelo <= 10000 for s in specs)


def test_misma_semilla_con_chunked_prefill_on_off(tmp_path):
    specs = cargar_experimento(_yaml(tmp_path, BASICO)).expandir()
    por_punto: dict[tuple, set[int]] = {}
    for s in specs:
        por_punto.setdefault((s.l_in, s.concurrencia, s.repeticion), set()).add(s.semilla_prompts)
    assert all(len(semillas) == 1 for semillas in por_punto.values())
    # y distinta entre repeticiones
    assert len({s.semilla_prompts for s in specs if s.l_in == 100 and s.concurrencia == 1}) == 2


def test_online_requiere_tasa(tmp_path):
    texto = BASICO.replace("modo_carga: offline", "modo_carga: online").replace(
        "lotes_por_corrida: 3", "num_peticiones: 10")
    with pytest.raises(ErrorConfig, match="tasa_llegada"):
        cargar_experimento(_yaml(tmp_path, texto))


def test_lotes_o_num_peticiones_exclusivos(tmp_path):
    texto = BASICO.replace("lotes_por_corrida: 3", "lotes_por_corrida: 3, num_peticiones: 5")
    with pytest.raises(ErrorConfig, match="exactamente uno"):
        cargar_experimento(_yaml(tmp_path, texto))


def test_campo_servidor_desconocido(tmp_path):
    texto = BASICO + "servidor: {tensor_parallel: 2}\n"
    with pytest.raises(ErrorConfig, match="desconocidos"):
        cargar_experimento(_yaml(tmp_path, texto))


@pytest.mark.parametrize("archivo", sorted(p.name for p in CONFIG.glob("*.yaml") if p.name != "servidor_base.yaml"))
def test_configs_del_repositorio_cargan(archivo):
    exp = cargar_experimento(CONFIG / archivo)
    specs = exp.expandir()
    assert specs, f"{archivo} no genera corridas"
    assert len({(s.servidor.config_id, s.l_in, s.l_out, s.concurrencia, s.tasa_llegada, s.repeticion)
                for s in specs}) == len(specs)
