#!/usr/bin/env python3
"""Mide prefill y decode de un modelo servido por Ollama y lo guarda todo en JSON.

Solo biblioteca estándar (Python 3.8+). Usa la API nativa /api/generate en streaming:
- lado cliente: TTFT (hasta el primer fragmento con texto), ITL entre fragmentos, latencia total;
- lado servidor (lo reporta Ollama, en ns): prompt_eval_count/duration (prefill),
  eval_count/duration (decode), load_duration.

Cada solicitud lleva un prefijo único para que Ollama no reutilice la caché KV de la anterior
(si no, el prefill de las repeticiones saldría artificialmente corto).

Uso:
  python3 medir_ollama.py --modelo qwen3-coder:30b --entradas 128,512,2048 --salida 128 --reps 5
Escribe <dir>/datos.json (todo) y <dir>/resumen.json (agregados por punto).
"""
import argparse, json, os, platform, socket, statistics, subprocess, time, urllib.request
from datetime import datetime

TEXTO = ("El rendimiento de la inferencia de modelos de lenguaje depende de la fase de prefill, "
         "que procesa el contexto completo en paralelo, y de la fase de decode, que genera un token "
         "tras otro y suele estar limitada por el ancho de banda de memoria. ")


def http_json(url, datos=None, timeout=30):
    req = urllib.request.Request(url, data=None if datos is None else json.dumps(datos).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gpu_mem():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,name",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        usado, total, nombre = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"usado_mib": int(usado), "total_mib": int(total), "nombre": nombre}
    except Exception as e:
        return {"error": str(e)}


def loadavg():
    try:
        return [float(x) for x in open("/proc/loadavg").read().split()[:3]]
    except Exception:
        return None


def prompt_aprox(n_palabras, i):
    palabras = TEXTO.split()
    cuerpo = " ".join(palabras[k % len(palabras)] for k in range(n_palabras))
    return "Solicitud %d-%d. %s\nResume el texto anterior:" % (i, time.time_ns() % 10**9, cuerpo)


def una_solicitud(base, modelo, prompt, salida, num_ctx, timeout):
    cuerpo = {"model": modelo, "prompt": prompt, "raw": True, "stream": True,
              "options": {"num_predict": salida, "temperature": 0, "seed": 0, "num_ctx": num_ctx}}
    req = urllib.request.Request(base + "/api/generate", data=json.dumps(cuerpo).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    marcas, final, error = [], None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for linea in r:
                if not linea.strip():
                    continue
                d = json.loads(linea)
                if d.get("response"):
                    marcas.append(time.perf_counter() - t0)
                if d.get("error"):
                    error = d["error"]
                if d.get("done"):
                    final = d
    except Exception as e:
        error = repr(e)
    total = time.perf_counter() - t0
    res = {"ok": error is None and final is not None, "error": error, "latencia_s": total,
           "ttft_s": marcas[0] if marcas else None, "fragmentos": len(marcas),
           "itl_s": [b - a for a, b in zip(marcas, marcas[1:])]}
    if final:
        ns = lambda k: final.get(k, 0) / 1e9
        res.update({
            "prompt_tokens": final.get("prompt_eval_count"), "salida_tokens": final.get("eval_count"),
            "done_reason": final.get("done_reason"),
            "servidor_load_s": ns("load_duration"), "servidor_prefill_s": ns("prompt_eval_duration"),
            "servidor_decode_s": ns("eval_duration"), "servidor_total_s": ns("total_duration")})
        if final.get("eval_count", 0) > 1 and final.get("eval_duration"):
            res["servidor_tpot_s"] = ns("eval_duration") / final["eval_count"]
            res["servidor_decode_tok_s"] = final["eval_count"] / ns("eval_duration")
        if final.get("prompt_eval_count") and final.get("prompt_eval_duration"):
            res["servidor_prefill_tok_s"] = final["prompt_eval_count"] / ns("prompt_eval_duration")
    if res["ttft_s"] is not None and len(marcas) > 1:
        res["tpot_cliente_s"] = (marcas[-1] - marcas[0]) / (len(marcas) - 1)
    return res


def estad(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs_o = sorted(xs)
    p = lambda q: xs_o[min(len(xs_o) - 1, int(round(q * (len(xs_o) - 1))))]
    return {"n": len(xs), "media": statistics.mean(xs), "mediana": statistics.median(xs),
            "p90": p(0.9), "min": xs_o[0], "max": xs_o[-1],
            "desv": statistics.stdev(xs) if len(xs) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:11434")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--entradas", default="128,512,2048", help="palabras aprox. de prompt por punto")
    ap.add_argument("--salida", type=int, default=128, help="num_predict")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--calentamiento", type=int, default=1)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--dir", default=None)
    a = ap.parse_args()

    marca = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = a.dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados",
                              "%s-%s" % (a.modelo.replace(":", "_").replace("/", "_"), marca))
    os.makedirs(d, exist_ok=True)

    info_modelo = http_json(a.base + "/api/show", {"model": a.modelo})
    meta = {
        "herramienta": "medir_ollama.py", "inicio": datetime.now().isoformat(timespec="seconds"),
        "nodo": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(), "ollama_version": http_json(a.base + "/api/version").get("version"),
        "modelo": a.modelo, "modelo_detalles": info_modelo.get("details"),
        "parametros": {"entradas": a.entradas, "salida": a.salida, "reps": a.reps,
                       "calentamiento": a.calentamiento, "num_ctx": a.num_ctx, "temperature": 0, "raw": True},
        "gpu_antes": gpu_mem(), "loadavg_antes": loadavg()}
    datos = {"meta": meta, "carga_modelo": None, "puntos": []}
    guardar = lambda: json.dump(datos, open(os.path.join(d, "datos.json"), "w"), indent=2, ensure_ascii=False)
    print("[%s] resultados en %s" % (time.strftime("%T"), d), flush=True)

    # Carga del modelo (primera solicitud, cronometrada aparte)
    print("[%s] cargando modelo..." % time.strftime("%T"), flush=True)
    r = una_solicitud(a.base, a.modelo, "Hola", 1, a.num_ctx, a.timeout)
    r["gpu_despues"] = gpu_mem()
    datos["carga_modelo"] = r
    guardar()
    print("[%s] carga: ok=%s latencia=%.1f s load=%.1f s GPU=%s MiB" % (
        time.strftime("%T"), r["ok"], r["latencia_s"], r.get("servidor_load_s") or -1,
        r["gpu_despues"].get("usado_mib")), flush=True)
    if not r["ok"]:
        print("FALLO al cargar: %s" % r["error"], flush=True)
        return 1

    for n in [int(x) for x in a.entradas.split(",")]:
        punto = {"entrada_palabras": n, "salida": a.salida, "calentamiento": [], "solicitudes": []}
        datos["puntos"].append(punto)
        for i in range(a.calentamiento + a.reps):
            res = una_solicitud(a.base, a.modelo, prompt_aprox(n, i), a.salida, a.num_ctx, a.timeout)
            (punto["calentamiento"] if i < a.calentamiento else punto["solicitudes"]).append(res)
            print("[%s] in~%d #%d ok=%s prompt=%s out=%s ttft=%.3f s prefill=%.3f s tpot=%.1f ms" % (
                time.strftime("%T"), n, i, res["ok"], res.get("prompt_tokens"), res.get("salida_tokens"),
                res["ttft_s"] or -1, res.get("servidor_prefill_s") or -1,
                1000 * (res.get("servidor_tpot_s") or -0.001)), flush=True)
            guardar()
        sol = [s for s in punto["solicitudes"] if s["ok"]]
        punto["agregados"] = {k: estad([s.get(k) for s in sol]) for k in (
            "prompt_tokens", "salida_tokens", "ttft_s", "servidor_prefill_s", "servidor_prefill_tok_s",
            "servidor_tpot_s", "servidor_decode_tok_s", "tpot_cliente_s", "latencia_s")}
        punto["fallos"] = len(punto["solicitudes"]) - len(sol)
        guardar()

    meta["fin"] = datetime.now().isoformat(timespec="seconds")
    meta["gpu_fin"] = gpu_mem()
    meta["loadavg_fin"] = loadavg()
    guardar()
    resumen = {"meta": {k: meta[k] for k in ("modelo", "nodo", "slurm_job", "ollama_version", "inicio", "fin",
                                              "parametros", "gpu_fin", "loadavg_antes", "loadavg_fin")},
               "carga_modelo_s": datos["carga_modelo"]["latencia_s"],
               "puntos": [{"entrada_palabras": p["entrada_palabras"], "fallos": p["fallos"], **p["agregados"]}
                          for p in datos["puntos"]]}
    json.dump(resumen, open(os.path.join(d, "resumen.json"), "w"), indent=2, ensure_ascii=False)
    print("[%s] fin; resumen en %s" % (time.strftime("%T"), os.path.join(d, "resumen.json")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
