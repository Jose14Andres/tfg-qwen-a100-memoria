from tfgbench.promptgen import VOCAB_REGULAR, lote_prompts, prompt_ids


def test_longitud_exacta_y_rango():
    for l_in in (1, 128, 4096):
        ids = prompt_ids(l_in, semilla=7, indice=0)
        assert len(ids) == l_in
        assert all(0 <= t < VOCAB_REGULAR for t in ids)


def test_determinista():
    assert prompt_ids(256, 42, 3) == prompt_ids(256, 42, 3)


def test_prompts_distintos_por_indice_y_semilla():
    lote = lote_prompts(128, 50, semilla=1)
    assert len({tuple(p) for p in lote}) == 50
    assert prompt_ids(128, 1, 0) != prompt_ids(128, 2, 0)


def test_lote_por_tramos_coincide():
    completo = lote_prompts(64, 10, semilla=5)
    assert lote_prompts(64, 5, semilla=5) + lote_prompts(64, 5, semilla=5, desde=5) == completo
