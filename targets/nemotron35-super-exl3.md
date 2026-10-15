# Nemotron 3.5 Super, RTX 5090

- Model: `acrogenesis/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-EXL3-3bpw`
- Revision: `8cff1e817bd8f0a29d57b3f5ed5b89163670ff68`
- Weights: 42 safetensors files, 49,998,109,886 bytes. Text only; MTP weights are present but MTP is disabled.
- Engine: ExLlamaV3 `058b8630fd081f3ff46fce6246955b157c3a2e3b` plus the architecture and CPU activation patches in `patches/`.
- Hardware under test: RTX 5090 32 GB, Ryzen 9950X3D, 128 GB DDR5.
- Runtime: 131,072 total cache tokens, four batch slots, 2048-token prefill chunks, 27 GiB allocation budget, 1 GiB recurrent checkpoint budget.
- Output cache is reserved in 2048-token scheduling chunks, with requeueing for longer completions; generation is not capped to this size.
- Each eligible MoE layer: 192 GPU experts and 320 CPU experts; eight CPU worker threads.
- `EXL3_MOE_CPU_WIDE=1`: upstream adaptive CPU activation precision. The included patch also supports opt-in mode `2`, using the existing high/low two-row representation for every nonzero SIMD row (15-bit range); both modes were scored, and neither reached the parity gate.
- `EXL3_INT8_GEMV=0`, `EXL3_MOE_CPU_SWAP=0`; no speculative decoding.
- Quality gate: top-1 agreement ≥ 0.99 and mean KL over reference top-20 ≤ 0.001 against the same EXL3 weights with all-GPU layer streaming.

Memory requirements are recorded from the new run, not inferred from file size. The container is capped at 56 GiB with no swap. This target does not claim support for another GPU or full source-model 1M context.
