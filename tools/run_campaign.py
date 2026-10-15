"""Run the quality gate and the complete recipe protocol against an idle server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--url', default='http://127.0.0.1:30080')
ap.add_argument('--out', required=True)
ap.add_argument('--config', required=True)
a = ap.parse_args()
root = Path(__file__).resolve().parents[1]
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
state = {'started': time.time(), 'status': 'running'}


def step(name, argv, timeout):
    state.update(stage=name, updated=time.time())
    (out/'state.json').write_text(json.dumps(state, indent=2)+'\n')
    with (out/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable, *argv], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)


try:
    step('quality', ['tools/score_ref_panel.py', '--url', a.url, '--panel',
                    'reference/nemotron35-exl3.json', '--out', str(out/'quality.json')], 1800)
    step('quality-gate', ['tools/check_quality.py', str(out/'quality.json')], 30)
    step('needle', ['tools/needle.py', '--url', a.url, '--tokens', '120000', '--depth', '0.8',
                   '--out', str(out/'needle.json')], 1800)
    needle = json.loads((out/'needle.json').read_text())
    if not needle['pass']:
        raise RuntimeError('120K retrieval check failed; investigate before speed testing')
    step('sweep', ['tools/sweep.py', '--url', a.url, '--card', 'rtx-5090-32gb',
                  '--config', a.config, '--out', str(out/'sweep.json')], 10800)
    state.update(status='complete', finished=time.time())
except Exception as error:
    state.update(status='failed', error=str(error), finished=time.time())
    raise
finally:
    (out/'state.json').write_text(json.dumps(state, indent=2)+'\n')
