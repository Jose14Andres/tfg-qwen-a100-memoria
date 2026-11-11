#!/usr/bin/env python3
"""Junta todos los ollama/resultados/*/resumen.json en resumen_global.json y una tabla Markdown.

Uso: python3 resumir.py   (desde cualquier sitio; escribe en ollama/)
"""
import glob, json, os

D = os.path.dirname(os.path.abspath(__file__))
filas = []
for f in sorted(glob.glob(os.path.join(D, "resultados", "*", "resumen.json"))):
    r = json.load(open(f))
    for p in r["puntos"]:
        med = lambda k: None if p.get(k) is None else p[k]["mediana"]
        filas.append({
            "corrida": os.path.basename(os.path.dirname(f)), "motor": "ollama", "modelo": r["meta"]["modelo"],
            "nodo": r["meta"]["nodo"], "inicio": r["meta"]["inicio"], "carga_modelo_s": r["carga_modelo_s"],
            "gpu_usado_mib": r["meta"]["gpu_fin"].get("usado_mib"),
            "loadavg_1m_antes": (r["meta"].get("loadavg_antes") or [None])[0],
            "entrada_palabras": p["entrada_palabras"], "n": (p.get("ttft_s") or {}).get("n"), "fallos": p["fallos"],
            "prompt_tokens": med("prompt_tokens"), "salida_tokens": med("salida_tokens"),
            "ttft_ms": None if med("ttft_s") is None else 1000 * med("ttft_s"),
            "prefill_ms": None if med("servidor_prefill_s") is None else 1000 * med("servidor_prefill_s"),
            "prefill_tok_s": med("servidor_prefill_tok_s"),
            "tpot_ms": None if med("servidor_tpot_s") is None else 1000 * med("servidor_tpot_s"),
            "decode_tok_s": med("servidor_decode_tok_s"), "latencia_s": med("latencia_s")})

json.dump({"descripcion": "medianas por punto; ttft = cliente, prefill/tpot = reportados por Ollama",
           "filas": filas}, open(os.path.join(D, "resumen_global.json"), "w"), indent=2, ensure_ascii=False)

cols = ["modelo", "prompt_tokens", "ttft_ms", "prefill_ms", "prefill_tok_s", "tpot_ms", "decode_tok_s", "latencia_s", "n", "fallos"]
fmt = lambda v: "-" if v is None else (("%.1f" % v) if isinstance(v, float) else str(v))
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(fmt(x[c]) for c in cols) + " |" for x in filas]
open(os.path.join(D, "tabla_ollama.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
