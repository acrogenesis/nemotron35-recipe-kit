# Parity measurements

Every row compares the same EXL3 weights with an independent all-GPU layer-streamed reference. The matched reference uses cached prefill and single-token decode. All failures are retained; the thresholds remain top-1 agreement >=99% and mean top-20 KL <=0.001.

| Server mode | Reference schedule | Matches / 300 | Mean top-20 KL | Gate |
|---|---|---:|---:|---|
| Initial adaptive CPU | Full sequence | 295 | 0.0008926502 | Failed |
| Initial adaptive CPU | Cached decode | 294 | 0.0006561110 | Failed |
| Forced-wide CPU | Cached decode | 295 | 0.0005988483 | Failed |
| Final adaptive CPU, used for speed | Cached decode | 296 | 0.0016589692 | Failed |

The reference schedule was matched before the precision experiments. The matched panel remained unchanged afterward. The source used for that panel is included here; the current builder only fixes a shadowed variable in its timing output.

The added `EXL3_MOE_CPU_WIDE=2` mode forces the existing two-row high/low activation representation for nonzero SIMD rows. Its six original-mode and 18 forced-mode kernel checks passed, but the full-model gate still failed. The final speed run uses upstream adaptive mode (`1`). No speed claim is made for forced-wide mode. The small panel shows run-to-run variation; these individual scores do not establish an accuracy improvement from mode `2`.

This is numerical runtime parity on six short fixed sequences. The model card reports the separate BF16-versus-EXL3 quantization-loss test. The complete speed campaign is explicitly diagnostic and does not establish quality parity with GPU-only inference.
