# Resumen prefill / decode — trabajo piloto-mig-0.6b-20261027-155013

## Entorno
```
(no existe)
```

## Prueba de humo (smoke.py)
```
(no existe)
```

## Offline (`vllm bench latency`, media de 10 iteraciones)

| entrada | lote | prefill T(out=1) ms | prefill tok/s | TPOT ms | decode tok/s por petición | decode tok/s total |
|---:|---:|---:|---:|---:|---:|---:|

## Online (`vllm serve` + `vllm bench serve`, dataset random, ignore-eos)

| entrada | salida | concurrencia | peticiones | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | ITL p99 ms | salida tok/s | total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1 | 8 | 18.9 | 20.2 | 8.3 | 11.0 | 118.5 | 237.1 |
| 128 | 128 | 4 | 8 | 51.3 | 54.7 | 9.7 | 11.1 | 400.5 | 801.0 |
| 1024 | 128 | 1 | 8 | 57.6 | 57.8 | 8.9 | 10.3 | 107.6 | 968.2 |
| 1024 | 128 | 4 | 8 | 183.9 | 221.4 | 12.4 | 14.9 | 288.9 | 2599.8 |

## Métricas del servidor (/metrics, sumas y conteos acumulados)
```
(no existe)
```
