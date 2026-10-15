"""Ensure a diagnostic sweep cannot silently become a quality-validated campaign."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CampaignGateTests(unittest.TestCase):
    def run_fixture(self, agreement, diagnostic=False, needle=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root/'tools'; tools.mkdir()
            shutil.copy2(ROOT/'tools/run_campaign.py', tools/'run_campaign.py')
            shutil.copy2(ROOT/'tools/check_quality.py', tools/'check_quality.py')
            # These explicit test fixtures never call a model or create measured results.
            payloads = {
                'score_ref_panel.py': {'positions': 300, 'top1_agreement': agreement, 'mean_kl_top20': .0001},
                'needle.py': {'pass': needle, 'fixture': True},
                'sweep.py': {'status': 'DONE', 'fixture': True},
            }
            for name, payload in payloads.items():
                (tools/name).write_text('import sys\nfrom pathlib import Path\n'
                    + 'Path(sys.argv[sys.argv.index("--out")+1]).write_text('
                    + repr(json.dumps(payload)) + ')\n')
            out = root/'out'
            command = [sys.executable, str(tools/'run_campaign.py'), '--out', str(out), '--config', 'TEST FIXTURE']
            if diagnostic:
                command.append('--diagnostic')
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            return result.returncode, json.loads((out/'state.json').read_text()), (out/'sweep.json').exists()

    def test_default_stops_before_speed_on_failed_parity(self):
        code, state, speed = self.run_fixture(.98)
        self.assertNotEqual(code, 0)
        self.assertEqual(state['status'], 'failed')
        self.assertFalse(state['quality_gate_passed'])
        self.assertFalse(speed)

    def test_diagnostic_preserves_both_failed_gates(self):
        code, state, speed = self.run_fixture(.98, diagnostic=True, needle=False)
        self.assertEqual(code, 0)
        self.assertEqual(state['status'], 'complete_diagnostic')
        self.assertFalse(state['quality_gate_passed'])
        self.assertFalse(state['needle_passed'])
        self.assertTrue(speed)

    def test_passing_standard_campaign(self):
        code, state, speed = self.run_fixture(1.0)
        self.assertEqual(code, 0)
        self.assertEqual(state['status'], 'complete')
        self.assertTrue(state['quality_gate_passed'])
        self.assertTrue(state['needle_passed'])
        self.assertTrue(speed)


if __name__ == '__main__':
    unittest.main()
