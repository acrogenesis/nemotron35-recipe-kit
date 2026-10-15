# Nemotron 3.5 Super recipe kit

Run the text component of Nemotron 3.5 Super 120B-A12B EXL3 on an RTX 5090 with CPU expert offloading, then measure it with a Nemotron adaptation of [Sybil's recipe protocol](https://github.com/sybil-solutions/local-ai-recipe-kit).

This kit uses **ExLlamaV3 with the included architecture patch**. It includes a small streaming API server, an independent reference-panel builder, quality checks, a long-context retrieval check, concurrent speed tests and memory/power measurements. It does not enable vision or MTP.

The target is a 32 GB RTX 5090, Ryzen 9950X3D and 128 GB RAM. The 50 GB weight files use VRAM plus RAM. [Exact settings](targets/nemotron35-super-exl3.md) · [Measurement protocol and adaptations](PROTOCOL.md).

## Build and download

Use Linux, Podman with NVIDIA CDI configured, Python 3, `bc`, and the Hugging Face CLI. The image builds CUDA kernels for SM120. The base image and engine source are pinned; the container is built locally, not downloaded from an unpublished image tag.

```bash
hf download acrogenesis/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-EXL3-3bpw \
  --revision 8cff1e817bd8f0a29d57b3f5ed5b89163670ff68 --local-dir ./models/nemotron35
timeout 3600 podman build --memory 16g -t localhost/nemotron35-recipe-kit:local .
```

## Serve

```bash
mkdir -p work/server
timeout 30 podman run -d --name nemotron35-recipe \
  --device nvidia.com/gpu=all --security-opt label=disable \
  --memory 56g --memory-swap 56g --shm-size 4g --ulimit memlock=-1:-1 \
  -p 127.0.0.1:30080:30000 \
  -v "$PWD/models/nemotron35:/models:ro" -v "$PWD/work/server:/results:rw" \
  localhost/nemotron35-recipe-kit:local
```

Wait for `curl --max-time 5 http://127.0.0.1:30080/health` to return `ok`. The endpoint is `http://127.0.0.1:30080/v1`, model `nemotron35-exl3`. The host port is bound to loopback. The server supports text chat, streaming and greedy sampling; it is a benchmark/reference server, not a complete OpenAI API implementation. No output cap is applied when the client omits it; the 131,072-token context remains the hard bound.

`timeout 30 podman stop nemotron35-recipe` stops it.

## Repeat the measurements

Stop the model server before building the reference panel. The reference streams one complete layer at a time through the GPU; the speed server uses CPU experts. The same quantized weights are compared in both paths.

```bash
timeout 1800 podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  --memory 16g --memory-swap 16g --shm-size 4g \
  -v "$PWD/models/nemotron35:/models:ro" -v "$PWD/reference:/reference:rw" \
  --entrypoint python localhost/nemotron35-recipe-kit:local \
  /kit/tools/build_reference.py --model /models --out /reference/nemotron35-exl3.json
```

Start `tools/memwatch.sh work/memwatch.csv` **before** launching the server, and leave it running through the campaign. Record `tools/probe.py`, `tools/hw_bw.py`, and the local image digest. Then launch the server as above and run:

```bash
python tools/smoke.py
python tools/run_campaign.py --out work/campaign \
  --config 'Nemotron EXL3; RTX5090; CPU320/threads8; batch4; ctx131072; chunk2048; INT8=0; swap=0'
```

The campaign stops if runtime parity or the 120K retrieval check fails. Decode requests run to EOS, so the complete sweep takes longer than a capped microbenchmark. Do not run another model or a CPU/GPU build while measuring. The [protocol](PROTOCOL.md) explains the scoring thresholds, random-token adaptation, cache checks and remaining limits.

## Provenance

Weights: [Hugging Face release](https://huggingface.co/acrogenesis/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-EXL3-3bpw), freshly quantized from NVIDIA's BF16 release. The model card includes a separate BF16-versus-EXL3 quantization-loss measurement; runtime parity in this kit does not measure that loss again.

Engine: [ExLlamaV3](https://github.com/turboderp-org/exllamav3), revision `058b8630fd081f3ff46fce6246955b157c3a2e3b`, with the text-only Nemotron adapter in `patches/`. Benchmark tools are adapted from [Sybil's recipe kit](https://github.com/sybil-solutions/local-ai-recipe-kit) at `ef9782ce0c564be8162e34edcb6bb626f56901d3`. Original tool hashes, protocol and MIT notice are in `upstream/`. Source code uses the MIT license; model weights retain their own OpenMDW-1.1 license and are not included here.
