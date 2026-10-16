#!/usr/bin/env python3
"""Paired EXL3/llama.cpp HTTP benchmark using identical saved token IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def make_fixture(url, path):
    info = get(url + "/server_info")
    valid = info["valid_random_token_ids"]
    topics = ["how TCP congestion control works", "how databases implement transactions",
              "how compilers optimise loops"]
    rows = []
    for rep in range(3):
        for size in (128, 8192, 32768):
            text = (f"[benchmark case: nemotron35-gguf-20261015-{size}-{rep}]\n"
                    f"Write a detailed, well-structured explanation of {topics[rep]}. "
                    "Include worked examples, limitations and practical applications. "
                    "Aim for at least 2500 words.")
            rendered = post(url + "/tokenize", {"messages": [{"role": "user", "content": text}],
                            "chat_template_kwargs": {"enable_thinking": False}})
            tail = rendered["tokens"]
            assert len(tail) <= size, (size, len(tail))
            rng = random.Random(20261015 + size * 10 + rep)
            ids = [rng.choice(valid) for _ in range(size - len(tail))] + tail
            rows.append({"id": f"p{size}-r{rep}", "prompt_tokens": size, "rep": rep,
                         "input_ids": ids, "input_ids_sha256": digest(ids),
                         "chat_tail": rendered["prompt"], "chat_tail_ids": tail})
    warm = post(url + "/tokenize", {"messages": [{"role": "user", "content":
                "[warmup nemotron35-gguf] Explain what an operating system does."}],
                "chat_template_kwargs": {"enable_thinking": False}})["tokens"]
    fixture = {"schema": 1, "seed": 20261015, "context": 131072, "output_cap": 512,
               "thinking": False, "temperature": 0, "warmup_ids": warm,
               "description": "Three distinct prompts per length; random non-special-token prefix "
                              "followed by the same rendered chat suffix. Prompt IDs are identical "
                              "across runtimes. This is a capped throughput test, not the natural-EOS sweep.",
               "cases": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"Saved {len(rows)} cases to {path}", flush=True)


def generate(url, backend, ids, cap):
    if backend == "exl3":
        route = "/generate"
        payload = {"input_ids": ids, "stream": True,
                   "sampling_params": {"temperature": 0, "max_new_tokens": cap}}
    else:
        route = "/completion"
        payload = {"prompt": ids, "stream": True, "n_predict": cap, "temperature": 0,
                   "seed": 20261015, "cache_prompt": False, "repeat_penalty": 1,
                   "presence_penalty": 0, "frequency_penalty": 0, "return_tokens": True,
                   "timings_per_token": False}
    request = urllib.request.Request(url + route, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    last_token = None
    events = []
    output_ids = []
    chunks = []
    final = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            raw = raw[5:].strip()
            if raw == b"[DONE]":
                break
            row = json.loads(raw)
            now = time.perf_counter()
            if "error" in row:
                raise RuntimeError(row["error"])
            if backend == "exl3":
                meta = row.get("meta_info", {})
                is_token = meta.get("completion_tokens", 0) > 0 and not meta.get("finish_reason")
                chunks = [row.get("text", "")]
                if meta.get("finish_reason"):
                    final = row
            else:
                is_token = not row.get("stop", False) and bool(row.get("tokens") or row.get("content"))
                chunks.append(row.get("content", ""))
                output_ids.extend(row.get("tokens", []))
                if row.get("stop", False):
                    final = row
            if is_token:
                first = now if first is None else first
                last_token = now
            events.append({"seconds": now - started, "token_event": is_token})
    ended = time.perf_counter()
    if final is None or first is None:
        raise RuntimeError(f"Missing completion or tokens: {final}")
    if backend == "exl3":
        meta = final["meta_info"]
        n = meta["completion_tokens"]
        cached = meta.get("cached_tokens")
        finish = meta["finish_reason"]
        timings = None
    else:
        n = final.get("tokens_predicted", final.get("timings", {}).get("predicted_n"))
        # llama.cpp tokens_cached includes the completed request's populated slot;
        # it is not the number of reused input tokens. cache_prompt=false disables reuse.
        timings = final.get("timings", {})
        cached = final.get("n_prompt_tokens_cache", timings.get("cache_n"))
        finish = final.get("stop_type", "limit" if final.get("stopped_limit") else "stop")
    if not isinstance(n, int) or n < 2:
        raise RuntimeError(f"Invalid completion token count: {n}")
    if backend == "exl3" and cached != 0:
        raise RuntimeError(f"Unexpected EXL3 prompt reuse: {cached}")
    # The terminal event carries authoritative token counts. This definition
    # matches the recipe sweep and includes the final SSE event overhead.
    terminal = started + events[-1]["seconds"]
    return {"completion_tokens": n, "ttft_seconds": first - started,
            "decode_tok_s": (n - 1) / (terminal - first),
            "aggregate_tok_s": n / (terminal - first),
            "end_to_end_tok_s": n / (ended - started), "seconds": ended - started,
            "last_token_seconds": last_token - started, "finish": finish,
            "cached_prompt_tokens": cached, "text": "".join(chunks),
            "output_ids": output_ids, "server_timings": timings,
            "terminal_event": final, "events": events}


def run(args):
    fixture = json.loads(args.fixture.read_text())
    info = get(args.url + ("/server_info" if args.backend == "exl3" else "/props"))
    info.pop("valid_random_token_ids", None)
    if args.backend == "exl3":
        assert info["context_length"] == fixture["context"]
    else:
        assert info["default_generation_settings"]["n_ctx"] == fixture["context"], info
        assert info["total_slots"] == 1, info["total_slots"]
    out = {"schema": 1, "label": args.label, "backend": args.backend,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "RUNNING",
           "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
           "context": fixture["context"], "output_cap": fixture["output_cap"],
           "server_info": info, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps(out, indent=1) + "\n")
        temp.replace(args.out)

    save()
    try:
        warm = generate(args.url, args.backend, fixture["warmup_ids"], 64)
        out["warmup"] = warm
        print(f"Warm-up: {warm['decode_tok_s']:.2f} tok/s", flush=True)
        for case in fixture["cases"]:
            row = generate(args.url, args.backend, case["input_ids"], fixture["output_cap"])
            row.update({k: case[k] for k in ("id", "prompt_tokens", "rep", "input_ids_sha256")})
            out["runs"].append(row)
            save()
            print(f"{case['id']}: {row['decode_tok_s']:.2f} tok/s, "
                  f"TTFT {row['ttft_seconds']:.3f}s, output {row['completion_tokens']}", flush=True)
        out["summary"] = {}
        for size in sorted({r["prompt_tokens"] for r in out["runs"]}):
            rows = [r for r in out["runs"] if r["prompt_tokens"] == size]
            out["summary"][str(size)] = {k: {"median": statistics.median(r[k] for r in rows),
                "all": [r[k] for r in rows]} for k in ("decode_tok_s", "ttft_seconds", "end_to_end_tok_s")}
        out["status"] = "DONE"
    except BaseException as error:
        out["status"] = "FAILED"
        out["error"] = repr(error)
        raise
    finally:
        out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("fixture")
    make.add_argument("--url", default="http://127.0.0.1:30080")
    make.add_argument("--out", required=True, type=Path)
    bench = sub.add_parser("run")
    bench.add_argument("--backend", required=True, choices=["exl3", "llama"])
    bench.add_argument("--url", required=True)
    bench.add_argument("--label", required=True)
    bench.add_argument("--fixture", required=True, type=Path)
    bench.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "fixture":
        make_fixture(args.url, args.out)
    else:
        run(args)


if __name__ == "__main__":
    main()
