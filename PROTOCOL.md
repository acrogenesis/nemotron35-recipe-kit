# Nemotron recipe measurement

Based on Sybil's recipe kit at `ef9782ce0c564be8162e34edcb6bb626f56901d3`.
The original protocol and tool hashes are retained in `upstream/` with its MIT license.

1. Record the hardware probe, GPU/CPU activity, host memory, swap and container memory.
2. Build `reference/nemotron35-exl3.json` with `tools/build_reference.py`: six fixed language, code, math and Spanish sequences. Each layer runs entirely on the GPU, then unloads. Within each layer it prefills all but the last prompt token, then runs single-token forwards with a cache, matching the server's numerical schedule. This avoids CPU offloading and fits the 50 GB quantization on one card. Continuations are teacher forced; they are not greedy answers. This is a runtime parity check against the same quantization, separate from the BF16-versus-EXL3 quantization-loss measurement on the model card. `--schedule full` retains the full-sequence diagnostic path; its matrix and recurrent kernels differ from cached decode.
3. Before speed tests, run `tools/score_ref_panel.py` against the server. Required top-1 agreement: at least 0.99. Required mean KL over the reference top-20 support: at most 0.001. These thresholds were set before this campaign. Stop on failure by default. An explicitly requested `--diagnostic` run may collect speed after recording the failure; its campaign is labeled `complete_diagnostic`, never quality-validated.
4. Test a needle at about 120,000 actual input tokens, depth 0.8, with the 131,072-token context configured. Record the exact token count and answer; do not treat this single retrieval task as general long-context quality.
5. Warm up with 512 random tokens; exclude it from results.
6. Prefill 8,192 / 16,384 / 32,768 / 65,536 tokens: three fresh random-token prompts each. Stop reading after the first streamed token and cancel the request. Throughput is input tokens divided by HTTP time to first token. Report median, range and all samples.
7. Decode at one, two, three and four concurrent users: two rounds each, with the upstream topics and natural-end completions. Also test one user with a fresh 32K prefix, twice. No output length cap, thinking budget or loop stopper. Reaching the context limit is a failed run. All requests use greedy sampling and thinking disabled.
8. Aggregate decode is completion tokens divided by the interval from the earliest first token to the latest final stream event. Per-stream decode is `(tokens - 1) / (last - first)`. These are Sybil's formulas. Record finish reasons, outputs, prompt hashes, cache hits, first-token latency and GPU power.
9. Run the submission schema validator (this does not certify quality). Publish raw results, exact runtime versions, image digest, launch settings, memory trace and charts. Report failures and deviations alongside successful measurements.

## Adaptations from Sybil's tools

- Random prompts use token IDs actually present in Nemotron's tokenizer, excluding special tokens, instead of Qwen's fixed `[1000, 150000)` range. Nemotron has a smaller vocabulary.
- Chat uses the publisher's chat template via `/tokenize`, with thinking disabled. Topics and requested explanation style are unchanged.
- Every request gets a fresh nonce, including requests across concurrency tiers. Long-context rounds use different random prefixes. Cache hits fail the decode run.
- A barrier starts each decode cohort together, after tokenization. Worker errors become explicit failed results.
- Decode requests omit `max_new_tokens`; only the configured context bounds generation. The upstream tool divides a context-sized bound by concurrency. Neither approach intentionally shortens normal completions, but ours verifies natural EOS explicitly.
- ExLlama jobs reserve output cache in 2048-token chunks and requeue as needed. This allows concurrent uncapped requests; it is a scheduling budget, not an output limit.
- Both upstream adaptive CPU activation precision (`EXL3_MOE_CPU_WIDE=1`) and the added forced-wide mode (`2`) missed the predeclared parity gate. Both measured scores are retained. The diagnostic speed sweep uses the upstream adaptive mode. GPU INT8 GEMV remains off. The forced-wide patch is opt-in and is not claimed to fix runtime parity.
- The server exposes the small SGLang-compatible endpoint subset these tools need. It runs ExLlamaV3, not SGLang, and does not implement Sybil's model-specific expert-cache kernels.
- The reference panel is Nemotron-specific. `score_ref_panel.py` is unmodified; its `greedy_match` diagnostic is not a gate for fixed teacher-forced continuations.
- The needle tool measures the actual token count and names the correct served model.
- The initial host/GPU logger failed because `bc` was missing. That trace is explicitly unavailable; the independent cgroup trace remains valid. The corrected logger uses `awk` and decimal GB and was tested separately after the sweep. No GPU peak or host RAM drop is inferred from the failed trace.
- The published summary takes prefill/C1/C2/C3 from the initial full sweep and C4/C1@32K from a separately recorded repeat. The repeat followed a 99.38% host-CPU burst in the first C4 round and added container/other-host CPU accounting. Both raw runs remain available, and the summary names the source for each load; it does not choose the faster sample from each run.
- The best-known table starts empty for this new model/configuration. No unrelated Qwen or GLM speed is used as a threshold.

The desktop remains running. GPU display use and other host activity are recorded; unrelated applications are not killed. A material competing workload invalidates a speed run. Host swap may already be occupied; the model container has swap disabled and its own memory usage is recorded separately.
