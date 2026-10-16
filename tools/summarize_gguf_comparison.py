#!/usr/bin/env python3
"""Assemble the paired benchmark report without overwriting any source run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


LABELS = {
    "exl3": "EXL3 3-bit HQ",
    "iq2-base": "Unsloth UD-IQ2_XXS",
    "iq2-mtp": "Unsloth UD-IQ2_XXS + MTP",
    "iq3-base": "Unsloth UD-IQ3_XXS",
    "iq3-mtp": "Unsloth UD-IQ3_XXS + MTP",
}


def medians(data):
    result = {}
    for size in (128, 8192, 32768):
        rows = [r for r in data["runs"] if r["prompt_tokens"] == size]
        assert len(rows) == 3
        result[str(size)] = {key: {"median": statistics.median(r[key] for r in rows),
                                  "all": [r[key] for r in rows]}
                            for key in ("decode_tok_s", "ttft_seconds", "end_to_end_tok_s")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {"path_redaction": "Local home-directory prefix replaced with {HOME}; measurements unchanged.",
                  "sources": {}}
    provenance["tools_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).with_name(name) for name in (
            "compare_gguf.py", "summarize_gguf_comparison.py", "plot_gguf_comparison.py"))
    }

    def copy(source, target=None):
        raw = (args.source / source).read_bytes()
        public = raw.replace(str(Path.home()).encode(), b"{HOME}")
        target = target or source
        dest = args.out / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(public)
        provenance["sources"][target] = {"original_name": source,
            "original_sha256": hashlib.sha256(raw).hexdigest(),
            "published_sha256": hashlib.sha256(public).hexdigest()}

    def read(name):
        return json.loads((args.source / name).read_text())

    assert read("matrix-state.json")["status"] == "DONE"
    assert read("exl3-recheck-state.json")["status"] == "DONE"
    assert read("download-verified.json")["status"] == "DONE"
    fixtures = {c["id"]: c for c in read("fixture.json")["cases"]}
    models = {}
    for mode in LABELS:
        source = "exl3-recheck.json" if mode == "exl3" else mode + ".json"
        data = read(source)
        assert data["status"] == "DONE" and len(data["runs"]) == 9
        data["analysis_sources"] = [source]
        if mode == "iq2-base":
            repeat = read("iq2-base-recheck.json")
            assert repeat["status"] == "DONE" and len(repeat["runs"]) == 1
            replacement = repeat["runs"][0]
            assert replacement["id"] == "p8192-r0"
            data["runs"] = [replacement if r["id"] == replacement["id"] else r for r in data["runs"]]
            data["analysis_sources"].append("iq2-base-recheck.json")
            data["analysis_note"] = "First 8K case repeated because unrelated host CPU exceeded 25% for 17 consecutive samples; original retained."
        data["summary"] = medians(data)
        for row in data["runs"]:
            assert row["input_ids_sha256"] == fixtures[row["id"]]["input_ids_sha256"]
            assert row["completion_tokens"] == 512 and row["cached_prompt_tokens"] == 0
            if mode != "exl3":
                assert row["terminal_event"]["tokens_evaluated"] == row["prompt_tokens"]
                assert row["server_timings"]["prompt_n"] == row["prompt_tokens"]
                assert not row["terminal_event"]["truncated"]
        public = json.dumps(data, indent=1).replace(str(Path.home()), "{HOME}") + "\n"
        (args.out / (mode + ".json")).write_text(public)
        models[mode] = data
        copy(source, "raw/" + source)
    copy("exl3.json", "raw/exl3-initial.json")
    copy("iq2-base-recheck.json", "raw/iq2-base-recheck.json")
    for name in ("fixture.json", "fixture-iq2-recheck.json", "download-verified.json",
                 "model-manifest.json", "runtime-verified.json", "hardware.json", "gguf-metadata.json",
                 "matrix-state.json", "exl3-recheck-state.json", "exl3-launch.json"):
        copy(name)
    for pattern in ("*-launch.json", "*-verification.json", "*-tokenizer-check.json", "*-cgroup.csv", "*-memwatch.csv"):
        for file in args.source.glob(pattern):
            copy(file.name)
    for file in (args.source / "logs").glob("*-server.log"):
        copy("logs/" + file.name)
    for file in (args.source / "logs").glob("*-bench.log"):
        copy("logs/" + file.name)
    resources = {}
    for name in ("exl3", "exl3-recheck", "iq2-base", "iq2-base-recheck", "iq2-mtp", "iq3-base", "iq3-mtp"):
        rows = list(csv.DictReader((args.source / (name + "-cgroup.csv")).open()))
        gpu = list(csv.DictReader((args.source / (name + "-memwatch.csv")).open()))
        other = [float(r["other_host_cpu_percent"]) for r in rows]
        streak = longest = 0
        for value in other:
            streak = streak + 1 if value >= 25 else 0
            longest = max(longest, streak)
        resources[name] = {"peak_cgroup_bytes_during_measurement": max(int(r["memory_current"]) for r in rows),
            "peak_anonymous_bytes": max(int(r["anonymous_bytes"]) for r in rows),
            "peak_model_swap_bytes": max(int(r["swap_current"]) for r in rows),
            "oom_kills": max(int(r["oom_kill"]) for r in rows),
            "peak_total_gpu_mib_during_measurement": max(int(r["gpu_used_mib"]) for r in gpu),
            "other_host_cpu_median_percent": statistics.median(other),
            "other_host_cpu_max_percent": max(other),
            "longest_consecutive_samples_other_cpu_at_least25": longest}
        assert resources[name]["peak_model_swap_bytes"] == 0 and resources[name]["oom_kills"] == 0
    (args.out / "resources.json").write_text(json.dumps(resources, indent=2) + "\n")
    (args.out / "analysis-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary = {}
    for mode, data in models.items():
        timings = [r["server_timings"] or {} for r in data["runs"]]
        drafted = sum(t.get("draft_n", 0) for t in timings)
        accepted = sum(t.get("draft_n_accepted", 0) for t in timings)
        summary[mode] = {"label": LABELS[mode], "summary": data["summary"],
                         "sources": data["analysis_sources"],
                         "drafted_tokens": drafted, "accepted_draft_tokens": accepted,
                         "draft_acceptance_percent": 100*accepted/drafted if drafted else None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = lambda key: "\n".join("| " + LABELS[mode] + " | " + " | ".join(
        f"{data['summary'][str(n)][key]['median']:.2f}" for n in (128, 8192, 32768)) + " |"
        for mode, data in models.items())
    report = """# Nemotron 3.5 Super: EXL3 versus Unsloth GGUF

RTX 5090 32 GB, Ryzen 9950X3D, 128 GB DDR5. Same text model family, identical token inputs, 128K configured context, FP16 KV cache, thinking off and greedy decoding. Each cell is the median of three 512-token responses from a single user.

![Generation speed](throughput.png)

| Runtime / quantization | 128-token prompt | 8K prompt | 32K prompt |
|---|---:|---:|---:|
""" + table("decode_tok_s") + """

Output tokens per second, measured over HTTP from the first token to the final stream event: `(generated_tokens - 1) / elapsed`. The earlier kit sweep ran to natural EOS; its results use a different workload.

## Time to first token

| Runtime / quantization | 128-token prompt | 8K prompt | 32K prompt |
|---|---:|---:|---:|
""" + table("ttft_seconds") + """

Seconds after sending the request. All reported requests processed their full input with zero prompt-cache hits and generated 512 tokens without context truncation.

## Weights and runtime

EXL3 3-bit HQ is **50.0 GB**. Unsloth **UD-IQ2_XXS is 51.1 GB**, the closest download size; **UD-IQ3_XXS is 59.8 GB**. These are different mixed-precision quantizations of the BF16 model family. Precision labels alone do not make them equivalent.

GGUF runs use Unsloth's CUDA 13 llama.cpp build **b11643-mix-cfaf026**, source `3bf0e05945af8821ef53d6daaf80d2fb4f234bd6`, directly through its HTTP server. Eight CPU threads, batch/ubatch 2048, one slot, automatic CPU/GPU placement, 2048 MiB GPU margin, and MoE expert cache disabled. MTP runs use the embedded prediction head with up to three draft tokens. Automatic placement may change to accommodate that head. Exact arguments are in the launch JSON files.

GGUF revision: `unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF@75a5829055f69a0359085ae6bbd0a81b60bc05c6`. All six shards passed SHA-256 verification. EXL3 uses the existing pinned recipe image and model, with CPU320/threads8 and MTP disabled. Its existing [quality measurements](../2026-10-15-rtx5090/#measured-quality) remain separate from this speed comparison.

## Repeats and resource checks

An unrelated host CPU burst affected the first IQ2 8K request. That exact case was repeated and used regardless of speed. The original and repeat are retained under `raw/`; the 8K median is unchanged. EXL3 was repeated at the end because background CPU use increased during the session; the final repeat supplies the table and graph, and the initial baseline is retained.

All model processes recorded zero swap use and zero OOM kills. GPU figures in [resources.json](resources.json) include the desktop and cover the measurement windows. Cgroup memory includes file cache, whose accounting can span service restarts; these totals are not whole-model RAM requirements. Host swap was already occupied and is distinct from model-process swap.

Native llama.cpp SSE events can combine text when a UTF-8 character spans tokens. An initial checker incorrectly required 512 streamed token IDs; the corrected check uses the server's authoritative generated count. Timings were unchanged. Each run retains the final metadata, responses and client event timestamps. [File provenance](analysis-provenance.json) records hashes before and after local-path redaction. [Summary data](summary.json) includes MTP draft acceptance counts.

## Reproduce

Use the included [fixture](fixture.json) with the benchmark client:

```bash
python tools/compare_gguf.py run --backend llama --url http://127.0.0.1:30081 \\
  --label iq2-base --fixture results/2026-10-15-gguf/fixture.json --out run.json
```

The fixture preserves every input token ID, including the seeded random prefix used for long prompts. `--backend exl3` targets the recipe's `/generate` API; `llama` targets `/completion`. Check that each server exposes 131072 context before running. The `*-launch.json` files contain the pinned runtime and complete server arguments; replace local path placeholders with your model and CUDA-library paths.

Redraw the figure with `python tools/plot_gguf_comparison.py --results results/2026-10-15-gguf --out results/2026-10-15-gguf/throughput`.
"""
    (args.out / "README.md").write_text(report)
    print(table("decode_tok_s"))


if __name__ == "__main__":
    main()
