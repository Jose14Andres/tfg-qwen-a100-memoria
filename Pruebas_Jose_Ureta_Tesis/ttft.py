import time, json, sys, os, urllib.request
PORT=os.environ.get("PORT","18000"); MODEL=os.environ.get("MODEL","Qwen/Qwen3-0.6B")
IN=int(sys.argv[1]) if len(sys.argv)>1 else 512
OUT=int(sys.argv[2]) if len(sys.argv)>2 else 128
body=json.dumps({"model":MODEL,"prompt":"la inferencia de modelos de lenguaje "*(IN//6+1),
 "max_tokens":OUT,"temperature":0,"ignore_eos":True,"stream":True}).encode()
req=urllib.request.Request("http://127.0.0.1:%s/v1/completions"%PORT,data=body,
 headers={"Content-Type":"application/json"})
t0=time.perf_counter(); tf=None; tl=None; n=0; itl=[]
with urllib.request.urlopen(req) as r:
    for raw in r:
        s=raw.decode().strip()
        if not s.startswith("data: ") or s=="data: [DONE]": continue
        now=time.perf_counter(); d=json.loads(s[6:])
        if not d.get("choices") or not d["choices"][0].get("text"): continue
        if tf is None: tf=now
        else: itl.append((now-tl)*1e3)
        tl=now; n+=1
        print("\rTTFT %.1f ms | tokens %d | TPOT %.2f ms"%((tf-t0)*1e3,n,(tl-tf)*1e3/max(n-1,1)),end="",flush=True)
itl.sort()
print("\nTTFT %.1f ms | tokens %d | TPOT %.2f ms | ITL p50 %.2f | ITL p99 %.2f"%(
 (tf-t0)*1e3,n,(tl-tf)*1e3/max(n-1,1),itl[len(itl)//2],itl[int(len(itl)*0.99)]))
