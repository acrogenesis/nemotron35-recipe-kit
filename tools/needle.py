#!/usr/bin/env python3
"""Long-context needle check against an OpenAI-compatible server: filler text of ~N tokens with one fact inserted at a
given depth, then a question about it (chat, thinking off, temperature 0, completion runs to EOS).

  python3 tools/needle.py --url http://127.0.0.1:30100 --tokens 131072 --depth 0.6
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("river mountain lantern harbor orchard meadow copper violet thunder glacier canyon ember willow falcon "
         "compass quartz maple tide prairie beacon cobalt saffron marble lagoon summit basalt heron juniper").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--depth", type=float, default=0.6)
    ap.add_argument("--model", default="nemotron35-exl3")
    ap.add_argument("--out")
    a = ap.parse_args()
    rng = random.Random(7)
    # Fit to measured tokenizer length, with room for a natural answer.
    code = f"{rng.randrange(1000, 9999)}-{rng.randrange(1000, 9999)}"
    needle = f"Important: the vault access code for the Kestrel project is {code}."
    sents = [" ".join(rng.choice(WORDS) for _ in range(12)).capitalize()+"."
             for _ in range(a.tokens // 8)]
    def payload_for(n):
        items = list(sents[:n]); items.insert(int(n*a.depth), needle)
        msg = " ".join(items)+"\n\nQuestion: What is the vault access code for the Kestrel project? Reply with the code only."
        return {"model": a.model, "messages": [{"role": "user", "content": msg}], "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
    lo, hi = 1, len(sents)
    while lo < hi:
        mid = (lo+hi+1)//2
        req = urllib.request.Request(a.url+"/tokenize", data=json.dumps(payload_for(mid)).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            n_tokens = len(json.loads(r.read())["tokens"])
        if n_tokens <= a.tokens: lo = mid
        else: hi = mid-1
    payload = payload_for(lo)
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    ans = out["choices"][0]["message"]["content"]
    res = {"prompt_tokens": out["usage"]["prompt_tokens"], "depth": a.depth, "code": code, "answer": ans,
           "pass": code in (ans or ""), "seconds": round(dt, 1)}
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
