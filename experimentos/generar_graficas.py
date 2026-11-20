#!/usr/bin/env python3
"""Genera las figuras de la tesis a partir de los datos medidos en ~/tfg-jose-ureta.

Uso:  graficas/.venv/bin/python graficas/generar_graficas.py
Salida: graficas/figuras/*.png (300 ppp) y *.pdf (vectorial), graficas/datos_graficas.json
        (cada número dibujado, con su fichero de origen).

Solo lee; no modifica ningún resultado. Los valores derivados (no medidos directamente)
se marcan como tales en el JSON y en el pie de la figura.
"""
import csv
import json
import statistics as st
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "graficas" / "figuras"
SALIDA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- estilo
# Paleta categórica validada (orden fijo, nunca cíclico) + neutros.
AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIS, GRIS_CLARO = "#8a8984", "#c9c8c2"
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "axes.labelcolor": TINTA_2,
    "axes.edgecolor": GRIS_CLARO,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.7,
    "xtick.color": TINTA_2,
    "ytick.color": TINTA_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "hatch.color": "white",
    "hatch.linewidth": 1.2,
    "pdf.fonttype": 42,
})


def num(x, d=1):
    """Número con coma decimal (convención en español)."""
    return f"{x:.{d}f}".replace(".", ",")


def eje_coma(ax, eje="y", d=None):
    def f(v, _):
        if d is None:
            return num(v, 0) if abs(v - round(v)) < 1e-9 else num(v, 1)
        return num(v, d)
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(FuncFormatter(f))


DATOS = {"descripcion": "Valores exactos dibujados en cada figura. 'derivado' = calculado a partir de medidas, no medido directamente.",
         "generado": datetime.now().isoformat(timespec="seconds"), "figuras": {}}


def guardar(fig, nombre, titulo, fuente, datos, nota=None):
    if nota:
        fig.text(0.01, 0.005, nota, fontsize=7.5, color=TINTA_2, ha="left", va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(SALIDA / f"{nombre}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    DATOS["figuras"][nombre] = {"titulo": titulo, "fuente": fuente, "nota": nota, "datos": datos}
    print("ok", nombre)


def barras(ax, x, h, color, ancho, **kw):
    """Barras finas con 2 px de separación (borde del color del fondo)."""
    return ax.bar(x, h, ancho, color=color, edgecolor="white", linewidth=1.0, **kw)


def hhmmss(s, fecha):
    return datetime.strptime(f"{fecha} {s}", "%Y-%m-%d %H:%M:%S")


def leer_monitor(f, col_mem):
    t, mem = [], []
    with open(f) as fh:
        for fila in csv.reader(fh):
            if not fila or not fila[0][:4].isdigit():
                continue
            try:
                t.append(datetime.strptime(fila[0].strip(), "%Y/%m/%d %H:%M:%S.%f"))
                mem.append(float(fila[col_mem].strip().split()[0]))
            except (ValueError, IndexError):
                continue
    return t, mem


def leer_monitor_cols(f):
    filas = []
    with open(f) as fh:
        for fila in csv.reader(fh):
            if not fila or not fila[0][:4].isdigit():
                continue
            try:
                filas.append((datetime.strptime(fila[0].strip(), "%Y/%m/%d %H:%M:%S.%f"),
                              *[float(c.strip().split()[0]) for c in fila[1:]]))
            except (ValueError, IndexError):
                continue
    return filas


# ================================================================ OLLAMA
OLL = RAIZ / "ollama" / "resultados"
CORRIDAS = [  # solo las corridas válidas (la del 11-nov de nemotron fue no válida: descarga parcial a CPU)
    ("qwen3-coder:30b", OLL / "qwen3-coder_30b-20261111-170123", AZUL, "job 33107, 11-nov"),
    ("nemotron-3-nano:30b", OLL / "nemotron-3-nano_30b-20261116-185849", NARANJA, "job 36268, 16-nov"),
]


def puntos_ollama(dirc):
    d = json.load(open(dirc / "datos.json"))
    out = []
    for p in d["puntos"]:
        sol = [s for s in p["solicitudes"] if s["ok"]]
        g = lambda k: [s[k] for s in sol]  # noqa: E731
        out.append({
            "entrada_palabras": p["entrada_palabras"],
            "prompt_tokens": st.median(g("prompt_tokens")),
            "n": len(sol),
            "ttft_ms": [v * 1000 for v in g("ttft_s")],
            "load_ms": [v * 1000 for v in g("servidor_load_s")],
            "prefill_ms": [v * 1000 for v in g("servidor_prefill_s")],
            "prefill_tok_s": g("servidor_prefill_tok_s"),
            "tpot_ms": [v * 1000 for v in g("servidor_tpot_s")],
            "decode_tok_s": g("servidor_decode_tok_s"),
            "latencia_s": g("latencia_s"),
        })
    return d, out


OLLDATA = {m: (c, et, *puntos_ollama(dirc)) for m, dirc, c, et in CORRIDAS}

# ---- F01: desglose del TTFT (load_duration + prefill + resto)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
reg = {}
for ax, (modelo, (color, etq, meta, pts)) in zip(axes, OLLDATA.items()):
    x = range(len(pts))
    load = [st.median(p["load_ms"]) for p in pts]
    pre = [st.median(p["prefill_ms"]) for p in pts]
    ttft = [st.median(p["ttft_ms"]) for p in pts]
    resto = [max(t - l - q, 0) for t, l, q in zip(ttft, load, pre)]
    barras(ax, x, pre, AZUL, 0.55, label="prefill (servidor)")
    barras(ax, x, load, NARANJA, 0.55, bottom=pre, label="load_duration (servidor)")
    barras(ax, x, resto, GRIS_CLARO, 0.55, bottom=[a + b for a, b in zip(pre, load)], label="resto (red, cliente)")
    for i, t in enumerate(ttft):
        ax.text(i, t + 60, f"{num(t, 0)} ms", ha="center", va="bottom", fontsize=8, color=TINTA)
    ax.set_xticks(list(x), [f"{int(p['prompt_tokens'])}" for p in pts])
    ax.set_xlabel("tokens del prompt")
    ax.set_title(modelo, fontsize=10)
    reg[modelo] = [{"prompt_tokens": p["prompt_tokens"], "ttft_cliente_ms_mediana": t, "prefill_ms_mediana": q,
                    "load_duration_ms_mediana": l, "resto_ms": r}
                   for p, t, q, l, r in zip(pts, ttft, pre, load, resto)]
axes[0].set_ylabel("TTFT medido en el cliente (ms)")
eje_coma(axes[0])
axes[0].set_ylim(0, max(max(st.median(p["ttft_ms"]) for p in v[3]) for v in OLLDATA.values()) * 1.18)
axes[1].legend(loc="upper left", fontsize=7.8)
fig.suptitle("Ollama: composición del tiempo hasta el primer token", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F01_ollama_desglose_ttft", "Ollama: composición del TTFT",
        ["ollama/resultados/*/datos.json"], reg,
        "Medianas de 5 solicitudes por punto, 128 tokens de salida, A100 40 GB. 'resto' = TTFT − prefill − load_duration (derivado).")

# ---- F02: rendimiento de prefill
fig, ax = plt.subplots(figsize=(6.0, 3.6))
reg = {}
for modelo, (color, etq, meta, pts) in OLLDATA.items():
    x = [p["prompt_tokens"] for p in pts]
    y = [st.median(p["prefill_tok_s"]) for p in pts]
    lo = [min(p["prefill_tok_s"]) for p in pts]
    hi = [max(p["prefill_tok_s"]) for p in pts]
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, y, "-o", color=color, lw=2, ms=6, mec="white", mew=1.5, label=f"{modelo} ({etq})")
    ax.annotate(f"{num(y[-1], 0)} tok/s", (x[-1], y[-1]), xytext=(0, 10 if color == AZUL else -14), textcoords="offset points",
                ha="center", va="center", fontsize=8, color=TINTA)
    reg[modelo] = [{"prompt_tokens": a, "prefill_tok_s_mediana": b, "min": c, "max": d}
                   for a, b, c, d in zip(x, y, lo, hi)]
ax.set_xlabel("tokens del prompt")
ax.set_ylabel("prefill (tokens/s)")
ax.set_ylim(0, 3450)
ax.set_xlim(0, 3300)
eje_coma(ax)
ax.legend(loc="lower right")
ax.set_title("Ollama: velocidad de prefill según la longitud del prompt")
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F02_ollama_prefill_toks", "Ollama: prefill tok/s vs prompt", ["ollama/resultados/*/datos.json"], reg,
        "Línea = mediana de 5 solicitudes; banda = mínimo–máximo. Valor reportado por Ollama (prompt_eval_duration).")

# ---- F03: TPOT
fig, ax = plt.subplots(figsize=(6.0, 3.6))
reg = {}
for modelo, (color, etq, meta, pts) in OLLDATA.items():
    x = [p["prompt_tokens"] for p in pts]
    y = [st.median(p["tpot_ms"]) for p in pts]
    lo = [min(p["tpot_ms"]) for p in pts]
    hi = [max(p["tpot_ms"]) for p in pts]
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, y, "-o", color=color, lw=2, ms=6, mec="white", mew=1.5, label=f"{modelo} ({etq})")
    for a, b in zip(x, y):
        ax.annotate(num(b, 2), (a, b), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.8, color=TINTA)
    reg[modelo] = [{"prompt_tokens": a, "tpot_ms_mediana": b, "min": c, "max": d,
                    "decode_tok_s_mediana": st.median(p["decode_tok_s"])}
                   for a, b, c, d, p in zip(x, y, lo, hi, pts)]
ax.set_xlabel("tokens del prompt")
ax.set_ylabel("TPOT (ms por token de salida)")
ax.set_ylim(0, 11)
ax.set_xlim(0, 3100)
eje_coma(ax)
ax.legend(loc="lower right")
ax.set_title("Ollama: tiempo por token de salida (decode)")
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F03_ollama_tpot", "Ollama: TPOT vs prompt", ["ollama/resultados/*/datos.json"], reg,
        "Mediana de 5 solicitudes, 128 tokens de salida; banda = mínimo–máximo. Valor reportado por Ollama (eval_duration).")

# ---- F04: tiempo de carga del modelo
cargas = [
    ("qwen3-coder:30b\n11-nov (job 33107)", json.load(open(OLL / "qwen3-coder_30b-20261111-170123/datos.json"))["carga_modelo"]["latencia_s"], AZUL, True),
    ("nemotron-3-nano:30b\n11-nov (job 33107)", None, NARANJA, False),
    ("nemotron-3-nano:30b\n16-nov (job 36268)", json.load(open(OLL / "nemotron-3-nano_30b-20261116-185849/datos.json"))["carga_modelo"]["latencia_s"], NARANJA, True),
]
d11 = json.load(open(OLL / "nemotron-3-nano_30b-20261111-170434/datos.json"))
cargas[1] = (cargas[1][0], (d11.get("carga_modelo") or {}).get("latencia_s") or 226.8, NARANJA, False)
fig, ax = plt.subplots(figsize=(6.0, 3.4))
reg = []
for i, (et, v, c, valida) in enumerate(cargas):
    b = barras(ax, i, v, c, 0.5, hatch=None if valida else "////")
    ax.text(i, v + 4, f"{num(v, 1)} s" + ("" if valida else "\nno válida*"), ha="center", va="bottom", fontsize=8, color=TINTA)
    reg.append({"corrida": et.replace("\n", " "), "carga_s": v, "valida": valida})
ax.set_xticks(range(len(cargas)), [c[0] for c in cargas], fontsize=8)
ax.set_ylabel("tiempo de carga (s)")
ax.set_ylim(0, 280)
eje_coma(ax)
ax.set_title("Ollama: carga del modelo en la GPU (pesos leídos de /home por NFS)")
fig.tight_layout(rect=(0, 0.06, 1, 1))
guardar(fig, "F04_ollama_carga_modelo", "Ollama: tiempo de carga del modelo", ["ollama/resultados/*/datos.json", "RESUMEN_TESIS.md"], reg,
        "*11-nov: Ollama leyó memoria libre obsoleta y dejó 10 de 53 capas en CPU; la medición posterior no es válida.")

# ---- F05: latencia por solicitud (dispersión) en Ollama
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
reg = {}
for ax, (modelo, (color, etq, meta, pts)) in zip(axes, OLLDATA.items()):
    for i, p in enumerate(pts):
        n = len(p["ttft_ms"])
        xs = [i - 0.12 + 0.24 * k / max(n - 1, 1) for k in range(n)]
        ax.scatter(xs, p["ttft_ms"], s=28, color=color, edgecolor="white", linewidth=1, zorder=3, label="TTFT cliente" if i == 0 else None)
        ax.scatter(xs, p["prefill_ms"], s=28, color=GRIS, marker="D", edgecolor="white", linewidth=1, zorder=3, label="prefill servidor" if i == 0 else None)
    ax.set_xticks(range(len(pts)), [str(int(p["prompt_tokens"])) for p in pts])
    ax.set_xlabel("tokens del prompt")
    ax.set_title(modelo, fontsize=10)
    reg[modelo] = [{"prompt_tokens": p["prompt_tokens"], "ttft_ms": p["ttft_ms"], "prefill_ms": p["prefill_ms"]} for p in pts]
axes[0].set_ylabel("ms (cada punto = 1 solicitud)")
eje_coma(axes[0])
axes[0].set_ylim(0, None)
axes[0].legend(loc="upper left")
fig.suptitle("Ollama: variabilidad solicitud a solicitud", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F05_ollama_dispersion_ttft", "Ollama: TTFT y prefill por solicitud", ["ollama/resultados/*/datos.json"], reg,
        "El prefill del servidor es casi constante; la dispersión del TTFT procede de load_duration (causa sin diagnosticar).")

# ================================================================ vLLM piloto (Qwen3-0.6B, MIG 1g.5gb)
PIL = RAIZ / "resultados" / "piloto-mig-0.6b-20261027-155013"
piloto = {}
for f in sorted(PIL.glob("serve_in*_out*_c*.json")):
    d = json.load(open(f))
    ent, conc = d["tfgbench"]["parametros"]["longitud_entrada"], d["tfgbench"]["parametros"]["concurrencia"]
    piloto[(ent, conc)] = d
ENTS, CONCS = sorted({k[0] for k in piloto}), sorted({k[1] for k in piloto})
CCOL = {1: AZUL, 4: NARANJA}

fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.3))
reg = []
ancho = 0.34
for j, (ax, met, tit) in enumerate(zip(axes, ["ttft", "tpot", "out"],
                                         ["TTFT p50 (ms)", "TPOT p50 (ms)", "salida agregada (tok/s)"])):
    for k, c in enumerate(CONCS):
        xs = [i + (k - 0.5) * ancho for i in range(len(ENTS))]
        if met == "ttft":
            ys = [piloto[(e, c)]["median_ttft_ms"] for e in ENTS]
            err = [[0] * len(ENTS), [piloto[(e, c)]["p99_ttft_ms"] - piloto[(e, c)]["median_ttft_ms"] for e in ENTS]]
        elif met == "tpot":
            ys = [piloto[(e, c)]["median_tpot_ms"] for e in ENTS]
            err = [[0] * len(ENTS), [piloto[(e, c)]["p99_tpot_ms"] - piloto[(e, c)]["median_tpot_ms"] for e in ENTS]]
        else:
            ys = [piloto[(e, c)]["output_throughput"] for e in ENTS]
            err = None
        barras(ax, xs, ys, CCOL[c], ancho, label=f"concurrencia {c}")
        if err:
            ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=TINTA_2, elinewidth=1, capsize=2.5)
        tops = [yy + (err[1][i] if err else 0) for i, yy in enumerate(ys)]
        for xx, yy, tp in zip(xs, ys, tops):
            ax.text(xx, tp * 1.02 + 0.5, num(yy, 1 if yy < 100 else 0), ha="center", va="bottom", fontsize=7.3, color=TINTA)
    ax.set_xticks(range(len(ENTS)), [f"{e}" for e in ENTS])
    ax.set_xlabel("tokens de entrada")
    ax.set_title(tit, fontsize=9.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    eje_coma(ax)
for e in ENTS:
    for c in CONCS:
        d = piloto[(e, c)]
        reg.append({"entrada": e, "salida": 128, "concurrencia": c, "solicitudes": d["completed"],
                    "ttft_p50_ms": d["median_ttft_ms"], "ttft_p99_ms": d["p99_ttft_ms"],
                    "tpot_p50_ms": d["median_tpot_ms"], "tpot_p99_ms": d["p99_tpot_ms"],
                    "itl_p99_ms": d["p99_itl_ms"], "salida_tok_s": d["output_throughput"],
                    "total_tok_s": d["total_token_throughput"]})
axes[2].legend(loc="upper right")
fig.suptitle("vLLM piloto: Qwen3-0.6B en MIG 1g.5gb (cliente tfgbench, 27-oct)", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F06_vllm_piloto_mig_metricas", "vLLM piloto MIG: TTFT, TPOT y throughput",
        ["resultados/piloto-mig-0.6b-20261027-155013/serve_*.json"], reg,
        "8 solicitudes por punto, 128 tokens de salida, ignore_eos. Barra = p50; bigote hasta p99.")

# ---- F07: distribución de ITL
fig, ax = plt.subplots(figsize=(6.4, 3.4))
reg = []
datos_box, etiquetas, colores = [], [], []
for e in ENTS:
    for c in CONCS:
        itl = [v * 1000 for fila in piloto[(e, c)]["itls"] for v in fila]
        datos_box.append(itl)
        etiquetas.append(f"in {e}\nc={c}")
        colores.append(CCOL[c])
        q = st.quantiles(itl, n=100)
        reg.append({"entrada": e, "concurrencia": c, "n_itl": len(itl), "p50_ms": st.median(itl),
                    "p1_ms": q[0], "p99_ms": q[98], "max_ms": max(itl)})
bp = ax.boxplot(datos_box, widths=0.5, patch_artist=True, whis=(1, 99), showfliers=True,
                flierprops=dict(marker="o", ms=2.5, mfc=GRIS, mec="none", alpha=0.5),
                medianprops=dict(color=TINTA, lw=1.5), whiskerprops=dict(color=TINTA_2), capprops=dict(color=TINTA_2))
for k, c in enumerate(colores):
    bp["boxes"][k].set(facecolor=c, edgecolor=c, alpha=0.9)
    bp["medians"][k].set(color=c, lw=3)
    for w in bp["whiskers"][2 * k:2 * k + 2] + bp["caps"][2 * k:2 * k + 2]:
        w.set(color=c, lw=1.4)
ax.set_xticks(range(1, len(etiquetas) + 1), etiquetas)
ax.set_ylabel("latencia entre tokens, ITL (ms)")
TOPE = 20
ax.set_ylim(0, TOPE)
for i, r in enumerate(reg):
    fuera = sum(v > TOPE for v in datos_box[i])
    if fuera:
        ax.annotate(f"{fuera} valores > {TOPE} ms\n(máx. {num(r['max_ms'], 1)} ms)", (i + 1, TOPE), xytext=(0, -4),
                    textcoords="offset points", ha="center", va="top", fontsize=7.3, color=TINTA_2)
eje_coma(ax)
ax.legend(handles=[Patch(color=AZUL, label="concurrencia 1"), Patch(color=NARANJA, label="concurrencia 4")], loc="lower right", ncol=2)
ax.set_title("vLLM piloto: distribución de la latencia entre tokens")
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F07_vllm_piloto_itl", "vLLM piloto MIG: distribución ITL",
        ["resultados/piloto-mig-0.6b-20261027-155013/serve_*.json"], reg,
        "Qwen3-0.6B, MIG 1g.5gb. Caja = p25–p75, línea = mediana, bigotes = p1–p99, puntos = fuera de ese rango. Eje recortado a 20 ms.")

# ================================================================ vLLM Qwen3-8B offline (job 30732)
B = RAIZ / "resultados" / "30732"
l1_128 = json.load(open(B / "lat_in128_out1_bs1.json"))
l1_512 = json.load(open(B / "lat_in512_out1_bs1.json"))
l129_128 = json.load(open(B / "lat_in128_out129_bs1.json"))
ttft128 = l1_128["percentiles"]["50"] * 1000
ttft512 = l1_512["percentiles"]["50"] * 1000
tpot128 = (l129_128["percentiles"]["50"] - l1_128["percentiles"]["50"]) * 1000 / 128
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw=dict(width_ratios=[2, 1.2]))
ax = axes[0]
barras(ax, [0, 1], [ttft128, ttft512], AZUL, 0.5)
for i, (v, n_tok) in enumerate([(ttft128, 128), (ttft512, 512)]):
    ax.text(i, v + 0.8, f"{num(v, 1)} ms\n({num(n_tok / v * 1000, 0)} tok/s)", ha="center", va="bottom", fontsize=8, color=TINTA)
ax.set_xticks([0, 1], ["128", "512"])
ax.set_xlabel("tokens de entrada")
ax.set_ylabel("latencia con 1 token de salida (ms)")
ax.set_ylim(0, 52)
ax.set_title("Prefill", fontsize=10)
eje_coma(ax)
ax = axes[1]
barras(ax, [0], [tpot128], AQUA, 0.5)
ax.text(0, tpot128 + 0.3, f"{num(tpot128, 2)} ms\n({num(1000 / tpot128, 1)} tok/s)", ha="center", va="bottom", fontsize=8, color=TINTA)
ax.set_xticks([0], ["entrada 128"])
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(0, 18)
ax.set_ylabel("ms por token de salida")
ax.set_title("Decode (derivado)", fontsize=10)
eje_coma(ax)
fig.suptitle("vLLM offline: Qwen3-8B en A100 40 GB (job 30732, 24-oct)", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 1))
guardar(fig, "F08_vllm_8b_offline", "vLLM bench latency Qwen3-8B",
        ["resultados/30732/lat_*.json"],
        {"prefill_in128_ms_p50": ttft128, "prefill_in512_ms_p50": ttft512,
         "lat_in128_out129_ms_p50": l129_128["percentiles"]["50"] * 1000,
         "tpot_in128_ms_derivado": tpot128, "iteraciones": len(l1_128["latencies"])},
        "vllm bench latency, lote 1, 10 iteraciones (p50). Decode derivado = (lat. out=129 − lat. out=1)/128. El barrido acabó en TIMEOUT (5 h).")

# ================================================================ arranques de vllm serve
# Hitos leídos de los logs (hora local). None = no ocurrió / no consta.
ARR = [
    {"id": "31157", "et": "Qwen3-8B · 27-oct\njob 31157 · compute-0-1", "fecha": "2026-10-27",
     "t0": "17:32:50", "banner": None, "engine": None, "carga_ini": None, "carga_fin": None, "fin": "18:17:56",
     "desenlace": "serve.log vacío; el script lo detuvo a los 45 min",
     "fuente": "tesis-8b-31157.out, tesis/resultados/31157/serve.log"},
    {"id": "31593", "et": "Qwen3-14B · 29-oct\njob 31593 · compute-0-2", "fecha": "2026-10-29",
     "t0": "16:24:20", "banner": "16:37:15", "engine": "16:57:00", "carga_ini": "17:00:42", "carga_fin": "17:03:51", "fin": "17:09:28",
     "desenlace": "pesos cargados; el script lo detuvo a los 45 min",
     "fuente": "tesis-14b-31593.out, tesis/resultados/31593/serve.log"},
    {"id": "base-0211", "et": "Qwen3-14B · 2-nov\nlínea base /home", "fecha": "2026-11-02",
     "t0": "09:08:51", "banner": "09:28:20", "engine": "09:54:46", "carga_ini": "09:59:16", "carga_fin": "10:04:02", "fin": "10:08:22",
     "desenlace": "TimeoutError de vLLM (VLLM_ENGINE_READY_TIMEOUT_S = 600 s)",
     "fuente": "docs/linea-base-home-20261102/{ensayo.out,serve.log}",
     "nota": "carga_ini derivado: 10:04:02 − 285,9 s ('Model loading took')"},
    {"id": "37415", "et": "Qwen3-14B · 19-nov\njob 37415 · compute-0-1", "fecha": "2026-11-19",
     "t0": "07:29:31", "banner": "07:55:19", "engine": None, "carga_ini": None, "carga_fin": None, "fin": "08:01:42",
     "desenlace": "el registro termina a los 32 min sin EngineCore (desenlace no registrado)",
     "fuente": "docs/arranque-14b-20261119/{arranque.out,serve.log,gpu_monitor.csv}"},
]
FASES = [("t0", "banner", "hasta el primer mensaje de vLLM (importación Python)", AZUL),
         ("banner", "engine", "API server → arranca EngineCore", NARANJA),
         ("engine", "carga_ini", "EngineCore → inicio de carga del modelo", AQUA),
         ("carga_ini", "carga_fin", "carga del modelo (pesos a GPU)", AMARILLO),
         ("carga_fin", "fin", "tras cargar pesos, sin quedar listo", GRIS)]
fig, ax = plt.subplots(figsize=(8.2, 3.6))
reg = []
for i, a in enumerate(ARR):
    y = len(ARR) - 1 - i
    t0 = hhmmss(a["t0"], a["fecha"])
    m = lambda k: (hhmmss(a[k], a["fecha"]) - t0).total_seconds() / 60 if a[k] else None  # noqa: E731
    fin = m("fin")
    fila = {"id": a["id"], "desenlace": a["desenlace"], "fuente": a["fuente"], "min_desde_t0": {}}
    hubo = False
    for k0, k1, _, c in FASES:
        x0, x1 = m(k0) if k0 != "t0" else 0.0, m(k1)
        if x0 is None or x1 is None:
            continue
        ax.barh(y, x1 - x0, left=x0, height=0.5, color=c, edgecolor="white", linewidth=1.0)
        fila["min_desde_t0"][f"{k0}->{k1}"] = [round(x0, 2), round(x1, 2)]
        hubo = True
    # tramo sin hitos hasta el final
    ultimo = max([m(k) for k in ("banner", "engine", "carga_ini", "carga_fin") if a[k]] or [0.0])
    if ultimo < fin and not (a["carga_fin"]):
        ax.barh(y, fin - ultimo, left=ultimo, height=0.5, color=GRIS_CLARO, edgecolor="white", linewidth=1.0, hatch="////")
        fila["min_desde_t0"]["sin_hitos"] = [round(ultimo, 2), round(fin, 2)]
    ax.text(fin + 0.8, y, f"{num(fin, 1)} min · {a['desenlace']}", va="center", fontsize=7.5, color=TINTA)
    fila["fin_min"] = fin
    if a.get("nota"):
        fila["nota"] = a["nota"]
    reg.append(fila)
ax.set_yticks(range(len(ARR)), [a["et"] for a in reversed(ARR)], fontsize=8)
ax.set_xlabel("minutos desde que se lanza vllm serve")
ax.set_xlim(0, 130)
ax.set_xticks(range(0, 70, 10))
ax.grid(axis="x", color=REJILLA)
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
leg = [Patch(color=c, label=e) for _, _, e, c in FASES] + [Patch(facecolor=GRIS_CLARO, hatch="////", edgecolor="white", label="sin nuevos mensajes en el log")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
ax.set_title("vLLM: ningún arranque de vllm serve llegó a quedar listo")
fig.tight_layout()
guardar(fig, "F09_vllm_arranques_cronologia", "Cronología de los arranques de vllm serve",
        [a["fuente"] for a in ARR], reg,
        "Hitos tomados de serve.log. Pesos (y el venv) leídos desde /home por NFS en los cuatro casos.")

# ---- F10: memoria GPU durante los arranques
MON = [("Qwen3-8B · 27-oct (31157)", RAIZ / "tesis/resultados/31157/gpu_monitor.csv", 2, "2026-10-27", "17:32:50", AZUL),
       ("Qwen3-14B · 29-oct (31593)", RAIZ / "tesis/resultados/31593/gpu_monitor.csv", 2, "2026-10-29", "16:24:20", NARANJA),
       ("Qwen3-14B · 19-nov (37415)", RAIZ / "docs/arranque-14b-20261119/gpu_monitor.csv", 1, "2026-11-19", "07:29:31", AQUA)]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
reg = {}
for et, f, col, fecha, t0s, c in MON:
    t, mem = leer_monitor(f, col)
    t0 = hhmmss(t0s, fecha)
    x = [(ti - t0).total_seconds() / 60 for ti in t]
    y = [v / 1024 for v in mem]
    ax.plot(x, y, color=c, lw=2, label=et, drawstyle="steps-post")
    reg[et] = {"muestras": len(x), "max_gib": max(y), "min_hasta_memoria_>0": next((xx for xx, yy in zip(x, y) if yy > 0), None),
               "duracion_min": x[-1]}
ax.axhline(40, color=GRIS_CLARO, lw=1, ls="--")
ax.text(1, 40.6, "capacidad A100: 40 GiB", fontsize=7.5, color=TINTA_2)
ax.set_xlabel("minutos desde que se lanza vllm serve")
ax.set_ylabel("memoria GPU usada (GiB)")
ax.set_ylim(-1, 44)
ax.set_xlim(0, 47)
eje_coma(ax)
ax.legend(loc="center left")
ax.set_title("vLLM: memoria de GPU durante el arranque")
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F10_vllm_arranques_memoria_gpu", "Memoria GPU durante el arranque de vllm serve",
        [str(m[1].relative_to(RAIZ)) for m in MON], reg,
        "nvidia-smi cada 1 s (31157, 31593) o 5 s (37415). Solo en 31593 llegaron pesos a la GPU (minuto ~37).")

# ---- F11: GPU durante el barrido offline 30732 (dos paneles, sin doble eje)
filas = leer_monitor_cols(B / "gpu_monitor.csv")  # t, util %, mem MiB, W, MHz
t0 = filas[0][0]
x = [(f[0] - t0).total_seconds() / 3600 for f in filas]
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
axes[0].plot(x, [f[2] / 1024 for f in filas], color=AZUL, lw=1.2)
axes[0].set_ylabel("memoria (GiB)")
axes[0].set_ylim(0, 42)
axes[1].plot(x, [f[1] for f in filas], color=NARANJA, lw=0.8)
axes[1].set_ylabel("utilización (%)")
axes[1].set_ylim(0, 105)
hitos = [("smoke.py", "2026-10-24 11:17:38"), ("in=128", "2026-10-24 11:48:32"), ("in=512", "2026-10-24 14:17:57")]
for ax in axes:
    for et, h in hitos:
        xh = (datetime.strptime(h, "%Y-%m-%d %H:%M:%S") - t0).total_seconds() / 3600
        ax.axvline(xh, color=GRIS, lw=0.8, ls=":")
        if ax is axes[0]:
            ax.text(xh + 0.03, 39, et, fontsize=7.5, color=TINTA_2, va="top")
    eje_coma(ax)
axes[1].set_xlabel("horas desde el inicio del monitor (24-oct, 11:17)")
axes[0].set_title("vLLM: uso de la GPU durante el barrido offline de Qwen3-8B (job 30732)")
util = [f[1] for f in filas]
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F11_vllm_8b_barrido_gpu", "GPU durante bench_pd.sh (30732)", ["resultados/30732/gpu_monitor.csv"],
        {"muestras": len(filas), "horas": x[-1], "mem_max_gib": max(f[2] for f in filas) / 1024,
         "fraccion_muestras_util>0": sum(u > 0 for u in util) / len(util), "hitos": hitos},
        "nvidia-smi cada 2 s. Cada fase de vllm bench latency recarga el modelo; la fase in=128 duró ~2,5 h (causa sin diagnosticar).")

# ---- F12: preparación del entorno (NFS)
prep = [
    ("31593 · 29-oct\nentorno (venv en /home)", (hhmmss("16:24:20", "2026-10-29") - hhmmss("16:21:48", "2026-10-29")).total_seconds() / 60, True, "tesis-14b-31593.out"),
    ("31157 · 27-oct\nentorno (venv en /home)", (hhmmss("17:32:50", "2026-10-27") - hhmmss("17:11:06", "2026-10-27")).total_seconds() / 60, True, "tesis-8b-31157.out"),
    ("32710 · 8-nov\ncopia hf-cache → /scratch", 459 / 60, True, "logs/fase1_32710.log"),
    ("33107 · 11-nov\ncopia hf-cache → /scratch", 414 / 60, True, "logs/fase1_33107.log"),
    ("33107 · 11-nov\ncopia venv (12 GB) → /scratch", 71, False, "RESUMEN_TESIS.md (1,2 GB de 12 GB a los 71 min)"),
]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
reg = []
for i, (et, v, completo, fu) in enumerate(prep):
    y = len(prep) - 1 - i
    ax.barh(y, v, height=0.5, color=AZUL if completo else GRIS_CLARO, edgecolor="white", linewidth=1.0, hatch=None if completo else "////")
    ax.text(v + 0.8, y, (f"{num(v, 1)} min" if completo else f"> {num(v, 0)} min, sin terminar (1,2 de 12 GB)"), va="center", fontsize=8, color=TINTA)
    reg.append({"paso": et.replace("\n", " "), "minutos": v, "completo": completo, "fuente": fu})
ax.set_yticks(range(len(prep)), [p[0] for p in reversed(prep)], fontsize=8)
ax.set_xlabel("minutos")
ax.set_xlim(0, 100)
ax.grid(axis="x", color=REJILLA)
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
eje_coma(ax, "x")
ax.set_title("Preparación del entorno: el cuello de botella es el NFS de /home")
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F12_preparacion_entorno_nfs", "Tiempos de preparación del entorno", [p[3] for p in prep], reg,
        "hf-cache de Qwen3-14B = 28 GB. Tiempos tomados de los logs de cada trabajo.")

# ---- F13: carga del nodo (loadavg 1 min al empezar cada prueba)
LA = [("30732\n24-oct", 37.20, "vLLM 8B offline\nTIMEOUT 5 h", "resultados/30732/entorno.txt"),
      ("31157\n27-oct", 134.39, "vLLM 8B\nno arrancó", "tesis/resultados/31157/entorno.txt"),
      ("31593\n29-oct", 147.24, "vLLM 14B\nno arrancó", "tesis/resultados/31593/entorno.txt"),
      ("33107\n11-nov", 333.15, "Ollama\nqwen3-coder ✓", "ollama/resultados/qwen3-coder_30b-20261111-170123/datos.json"),
      ("36268\n16-nov", 85.68, "Ollama\nnemotron ✓", "ollama/resultados/nemotron-3-nano_30b-20261116-185849/datos.json"),
      ("37415\n19-nov", 177.18, "vLLM 14B\nsin desenlace", "docs/arranque-14b-20261119/entorno.txt")]
fig, ax = plt.subplots(figsize=(7.2, 3.5))
reg = []
for i, (et, v, res, fu) in enumerate(LA):
    barras(ax, i, v, AZUL, 0.5)
    ax.text(i, v + 6, f"{num(v, 0)}\n{res}", ha="center", va="bottom", fontsize=7.3, color=TINTA)
    reg.append({"job": et.replace("\n", " "), "loadavg_1m": v, "resultado": res.replace("\n", " "), "fuente": fu})
ax.set_xticks(range(len(LA)), [l[0] for l in LA], fontsize=8)
ax.set_ylabel("loadavg (1 min) al empezar")
ax.set_ylim(0, 420)
eje_coma(ax)
ax.set_title("Carga del nodo compartido al inicio de cada prueba")
fig.tight_layout(rect=(0, 0.04, 1, 1))
guardar(fig, "F13_carga_nodo_loadavg", "loadavg del nodo por prueba", [l[3] for l in LA], reg,
        "loadavg del nodo completo (compartido con otros usuarios), leído al comenzar cada trabajo.")

# ================================================================ JSON con todo lo dibujado
(RAIZ / "graficas" / "datos_graficas.json").write_text(json.dumps(DATOS, ensure_ascii=False, indent=2, default=str))
print("datos_graficas.json escrito;", len(DATOS["figuras"]), "figuras")
