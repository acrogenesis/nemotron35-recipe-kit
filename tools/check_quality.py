"""Fail the campaign if the predeclared runtime parity thresholds are missed."""
import json
import sys

data = json.load(open(sys.argv[1]))
assert data['positions'] > 100
assert data['top1_agreement'] >= 0.99, data
assert data['mean_kl_top20'] <= 0.001, data
print('Runtime parity gate passed')
