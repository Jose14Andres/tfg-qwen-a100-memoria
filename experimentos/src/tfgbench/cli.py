"""Interfaz de línea de comandos: `python -m tfgbench <comando>`."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from .config import cargar_experimento
from .entorno import capturar_entorno


def _cmd_plan(args: argparse.Namespace) -> int:
    total = 0
    filas = []
    for ruta in args.experimentos:
        exp = cargar_experimento(ruta)
        specs = exp.expandir()
        total += len(specs)
        peticiones = sum(s.num_peticiones + s.num_warmup for s in specs)
        configs = Counter(s.servidor.config_id for s in specs)
        print(f"{exp.exp_id:7} {exp.nombre:28} corridas={len(specs):5d} "
              f"peticiones={peticiones:7d} configs={len(configs)} descartes={len(exp.descartes)}")
        for d in exp.descartes if args.verbose else []:
            print(f"        descartado: {d}")
        for s in specs:
            filas.append({"exp_id": s.exp_id, "config_id": s.servidor.config_id,
                          "chunked_prefill": s.servidor.chunked_prefill, "l_in": s.l_in, "l_out": s.l_out,
                          "concurrencia": s.concurrencia, "tasa_llegada": s.tasa_llegada,
                          "repeticion": s.repeticion, "num_peticiones": s.num_peticiones,
                          "num_warmup": s.num_warmup, "semilla_prompts": s.semilla_prompts})
    print(f"TOTAL corridas: {total}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        print(f"Matriz escrita en {args.csv}")
    return 0


def _cmd_entorno(args: argparse.Namespace) -> int:
    fila = capturar_entorno(args.proveedor, Path(args.lockfile) if args.lockfile else None,
                            permitir_sin_gpu=args.sin_gpu)
    print(json.dumps(fila, default=str, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tfgbench", description=__doc__)
    sub = p.add_subparsers(dest="comando", required=True)

    plan = sub.add_parser("plan", help="Expande la matriz experimental y la resume")
    plan.add_argument("experimentos", nargs="+", help="YAML de experimentos")
    plan.add_argument("--csv", help="Escribe la matriz completa en un CSV")
    plan.add_argument("-v", "--verbose", action="store_true", help="Muestra los puntos descartados")
    plan.set_defaults(func=_cmd_plan)

    ent = sub.add_parser("entorno", help="Captura e imprime la fila de entorno de esta máquina")
    ent.add_argument("--proveedor", required=True)
    ent.add_argument("--lockfile")
    ent.add_argument("--sin-gpu", action="store_true", help="Solo para desarrollo sin GPU")
    ent.set_defaults(func=_cmd_entorno)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
