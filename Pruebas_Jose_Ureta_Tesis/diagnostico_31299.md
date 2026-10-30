# Diagnóstico: la shell no responde en el salloc 31299 (28-oct-2026)

Objetivo de la sesión: arrancar `vllm serve` con Qwen3-8B en la A100 completa del salloc 31299 para medir
E1 (prefill) y E2 (concurrencia) a mano. **No se llegó a arrancar el servidor.**

## Sesión

| | |
|---|---|
| Trabajo | 31299, partición `gpu`, interactivo |
| Nodo | compute-0-2.hpc.cedia.edu.ec |
| GPU | `GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-09001e31-8224-a101-d97b-ab391ece7116)` (A100 completa, no MIG) |
| CPU asignadas | `76-79,204-207` (8 CPU, `cpuset.cpus` del trabajo) |

## Qué pasó

1. Los primeros comandos de la sesión (`hostname`, `nvidia-smi -L`, lectura de archivos) respondieron bien.
2. Después, **todos los comandos de shell se quedaron colgados**, incluso los triviales:
   - `cat /proc/loadavg; nvidia-smi ...; ss -ltn ...; ls ~/.cache/vllm/torch_compile_cache; df -h $HOME` → sin salida en 120 s
   - `timeout 10 cat /proc/loadavg; ...` → sin salida en 60 s (ni siquiera el `timeout` llegó a actuar, así que la shell no llegó a ejecutar nada)
   - `echo ok` → sin salida en 90 s
   Todos se detuvieron a mano; no quedó nada en marcha.
3. La lectura directa de archivos sí funcionaba, y con ella se tomaron estas medidas.

## Medidas del nodo (leídas directamente de /proc y /sys)

```
/proc/loadavg          163.88 166.92 175.17 144/4576 3188591
/proc/pressure/cpu     some avg10=21.88 avg60=21.38 avg300=21.50
/proc/pressure/io      some avg10=0.00 avg60=0.37 avg300=0.50   full avg10=0.00 avg60=0.35 avg300=0.48
/proc/pressure/memory  some avg10=0.00 avg60=0.00 avg300=0.00   full avg10=0.00 avg60=0.00 avg300=0.00
```

- Carga ~165 con solo 8 CPU propias: el nodo está muy cargado por otros trabajos.
- Presión de CPU ~21 %; disco y memoria sin presión. No parece un sistema de archivos colgado.
- No se pudo comprobar la memoria usada de la GPU ni si el puerto 8000 está libre (hace falta la shell).

## Causa probable (sin confirmar)

`~/.bashrc` hace esto en **cada** shell que se abre:

```bash
if [ -d "$HOME/spack" ]; then
    . $HOME/spack/share/spack/setup-env.sh
fi
spack load hwloc
```

`spack load` es lento (Python + base de datos de Spack con bloqueo) y en un nodo con esta carga puede dejar la
shell parada antes de ejecutar el primer comando. Encaja con que fallara hasta `echo ok`. No se pudo confirmar
porque no se pueden inspeccionar los procesos sin shell.

### Relación con el trabajo 31157 (27-oct, compute-0-1, carga ~134)

Los `sbatch` empiezan con `#!/bin/bash -l`, así que también ejecutan `spack load hwloc`:

- 16:54 el trabajo empieza (`tesis-8b-31157.err`)
- 17:11 primera línea del log (`entorno`): **~17 min** antes de ejecutar nada → compatible con la hipótesis
- 17:11 → 17:32 bloque de entorno (importar vllm, nvidia-smi): 21 min
- 17:32 → 18:17 `vllm serve` sin escribir nada en `serve.log`, GPU a 0 MiB → **sin explicar**

## Comprobación pendiente (desde tu terminal)

```bash
time bash -c 'echo ok'          # con .bashrc
time bash --norc -c 'echo ok'   # sin .bashrc
```

- **Solo el primero es lento** → se confirma. Arreglo propuesto (no aplicado): en `~/.bashrc` cambiar
  `spack load hwloc` por `[[ $- == *i* ]] && spack load hwloc` (solo en shells interactivas). Luego, arrancar el servidor.
- **Los dos son lentos** → el problema es el nodo. Cerrar este salloc y relanzar
  `sbatch --exclude=compute-0-1 tesis/a100_qwen3_8b.sh` (o excluir también compute-0-2 si sigue cargado).
