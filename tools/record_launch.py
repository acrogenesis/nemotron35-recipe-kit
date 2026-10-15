"""Record the local image identity and explicit launch settings; no host paths."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--image', default='localhost/nemotron35-recipe-kit:local')
ap.add_argument('--out', required=True)
a = ap.parse_args()
root = Path(__file__).resolve().parents[1]
image = json.loads(subprocess.check_output(['podman', 'image', 'inspect', a.image], timeout=30))[0]
digests = image.get('RepoDigests', [])
assert digests, 'Local built image has no recorded manifest digest'
record = {
    'image': digests[0], 'image_id': image['Id'], 'image_distribution': 'built locally from Containerfile; not published to a registry',
    'base_image': 'ghcr.io/0xsero/sglang-exl3-flashnext@sha256:86f673e630856c98c301b2731de3e1a0fe7639a3e12a7626abaea24a18e0eb5a',
    'args': ['python', '/kit/server/serve.py', '-m', '/models', '-cs', '131072', '-gs', '27',
             '-mcs', '320', '-mct', '8', '-ambs', '4', '-chunk_size', '2048', '-rcs', '1',
             '--host', '0.0.0.0', '--port', '30000'],
    'env': {'EXL3_INT8_GEMV': '0', 'EXL3_MOE_CPU_SWAP': '0', 'EXL3_MOE_CPU_WIDE': '1', 'TORCH_CUDA_ARCH_LIST': '12.0', 'MAX_JOBS': '4', 'PYTHONPATH': ''},
    'shm': '4g', 'gpu': 'nvidia.com/gpu=all', 'memory': '56g', 'memory_swap': '56g',
    'published_port': '127.0.0.1:30080:30000',
    'volumes': ['<model directory>:/models:ro', '<result directory>:/results:rw'],
    'model': 'acrogenesis/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-EXL3-3bpw',
    'model_revision': '8cff1e817bd8f0a29d57b3f5ed5b89163670ff68',
    'engine_revision': '058b8630fd081f3ff46fce6246955b157c3a2e3b',
    'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py', '.patch')
                      and not any(x in p.parts for x in ('work', '.git', '__pycache__'))},
}
Path(a.out).write_text(json.dumps(record, indent=2)+'\n')
print(record['image'])
