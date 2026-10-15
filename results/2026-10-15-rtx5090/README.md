# RTX 5090 results, 2026-10-15

**Status: complete diagnostic sweep; runtime parity failed.** Quality thresholds were retained. These speeds do not establish parity with GPU-only inference.

Hardware: RTX 5090 32 GB, Ryzen 9950X3D, 128 GB DDR5. Model: NVIDIA Nemotron 3.5 Super 120B-A12B, text-only EXL3 3-bit HQ, revision `8cff1e817bd8f0a29d57b3f5ed5b89163670ff68`. ExLlamaV3 1.6.2 with the included patches; adaptive CPU activations, 320 CPU experts/layer, eight CPU threads, four batch slots, 128K total cache, 2048-token prefill chunks. GPU INT8 GEMV, dynamic expert swapping, MTP and thinking are disabled.

## Generation

![Measured throughput](throughput.png)

| Load | Total output tok/s | Per-user tok/s |
|---|---:|---:|
| 1 user | 33.36 | 33.34 |
| 2 users | 36.25 | 18.12 |
| 3 users | 37.95 | 12.65 |
| 4 users | 40.25 | 10.15 |
| 1 user, 32K prompt | 33.64 | — |

Two rounds per load. Aggregate rate is total completion tokens divided by the span from the first token to the final stream event. Per-user rate is the mean of each stream's `(tokens - 1) / elapsed`, then the median of the two rounds. Streams may differ in response length and client-observed rate. The 32K row reports aggregate throughput for its single stream.

All 22 reported decode requests ended at stop tokens with zero prompt-cache hits. They produced 38,345 tokens total, ranging from 1,059 to 3,251 per response. The [combined report](summary.json) names each source: prefill/C1/C2/C3 from the [initial full sweep](sweep.json), C4 and C1@32K from the [targeted repeat](recheck-sweep.json). Both raw files retain all responses, timestamps and power readings; [verification.json](verification.json) records the integrity checks. These uncapped HTTP runs use a different workload from the earlier 256-token CLI microbenchmark.

## CPU-load recheck

The first C4 round in the initial sweep reached 99.38% total host CPU use, including 26 samples above 80%. C4 and C1@32K were repeated on the same running model and image with added container/other-host CPU tracking. The table and chart use those repeated loads; the original C4 median was 39.29 tok/s and original C1@32K median was 31.695 tok/s. This is a report assembled from the two explicitly named runs, not a replacement of the original raw sweep.

During the repeat, median container CPU use was 28.03% of total host capacity, and median other-host use was 4.29%. There was no sustained interval of at least five samples with other-host use at or above 25%. [Repeat resource summary](recheck-resources.json) · [CPU trace](cgroup-recheck.csv) · [host/GPU trace](memwatch-recheck.csv).

## Prompt processing

| Input tokens | Median tok/s | Range |
|---:|---:|---:|
| 8,192 | 1,112.5 | 1,018.3–1,114.7 |
| 16,384 | 1,200.5 | 1,113.5–1,224.5 |
| 32,768 | 1,227.4 | 1,207.8–1,238.5 |
| 65,536 | 1,245.4 | 1,222.1–1,250.4 |

Three fresh random-token inputs per size. Rates are input tokens divided by HTTP time to the first token; requests are then cancelled. The 512-token warm-up is excluded. This measures prompt-processing speed on random token sequences.

## Measured quality

Runtime parity: **296/300 top-1 matches (98.67%)**, mean KL on the reference top-20 support **0.00165897**. Both gates failed (required: >=99%, <=0.001). The reference uses the same EXL3 weights, all experts on GPU, streamed one layer at a time, with a matched cached-decode schedule. It covers six short teacher-forced sequences. [Scores](quality.json) · [earlier adaptive and forced-wide measurements](diagnostics/).

Long-context retrieval: **passed** at 119,983 actual input tokens, with the fact inserted at depth 80%. The model returned the expected code. This is one retrieval case; [needle.json](needle.json) contains the exact result.

The model card contains a separate BF16-versus-EXL3 quantization-loss measurement. This kit's panel measures runtime agreement within the same quantization.

## Resources and validation

Peak container memory was **46.32 GiB**, including file cache and startup; peak anonymous memory was 29.85 GiB. The separate host/GPU logger failed because `bc` was missing, so GPU peak and host RAM drop are unavailable. Container swap use, OOM events and OOM kills were all zero. The host already had occupied swap before launch. These figures describe this run, not a minimum-RAM guarantee. [Exact byte counts](resources.json) · [cgroup trace](cgroup.csv) · [failed host/GPU trace](diagnostics/host-gpu-trace-failed.csv).

The memory logger now uses `awk` with correct decimal-GB units and has been tested after the sweep. The follow-up trace measured a peak of 25.22 GiB total GPU memory during C4/32K only, including the desktop. It does not establish the full-sweep peak or startup host RAM drop. The valid cgroup summary uses exact byte counts. The desktop and unrelated applications remained running; host CPU activity is recorded in the cgroup trace.

The CPU patch passed six original-mode and 18 forced-mode upstream kernel checks. Three campaign tests verify failure/default/diagnostic behavior. Live API checks passed for invalid token IDs, four simultaneous requests, and disconnect cancellation. The submission schema validator passed; it does not certify quality.

[Launch settings and measured image digest](launch.json) · [hardware](probe.json) · [bandwidth](bandwidth.json) · [runtime](server-info.json) · [campaign state](state.json) · [submission](../../submissions/rtx-5090-32gb/2026-10-15-acrogenesis.json).

The image was built locally from pinned public sources. Its digest identifies the measured local build; it is not a registry-published image. The server uses ExLlamaV3 and the architecture patch, without Sybil's model-specific expert-cache kernels.

To redraw the chart from the committed data, run from the repository root after building the image:

```bash
mkdir -p work/charts
podman run --rm --entrypoint python \
  -v "$PWD:/kit:ro" -v "$PWD/work/charts:/charts:rw" \
  localhost/nemotron35-recipe-kit:local /kit/tools/plot_results.py \
  --sweep /kit/results/2026-10-15-rtx5090/summary.json \
  --quality /kit/results/2026-10-15-rtx5090/quality.json --out /charts/throughput
```
