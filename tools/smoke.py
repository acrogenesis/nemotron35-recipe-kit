"""Live server checks: validation, natural completion, cancellation, concurrency."""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:30080')
a = ap.parse_args()


def post(path, body):
    req = urllib.request.Request(a.url+path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


info = json.load(urllib.request.urlopen(a.url+'/server_info', timeout=30))
assert info['max_running_requests'] == 4, info
for invalid in [[info['vocab_size']], [-1], [], [True]]:
    try:
        post('/generate', {'input_ids': invalid})
    except urllib.error.HTTPError as e:
        assert e.code == 400, e
    else:
        raise AssertionError(f'Accepted invalid token IDs: {invalid}')


def chat(n):
    return post('/v1/chat/completions', {'messages': [{'role': 'user', 'content':
        f'Check {n}: What is 2 + 2? Reply with the number only.'}], 'temperature': 0})


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    replies = list(pool.map(chat, range(4)))
assert all(r['choices'][0]['finish_reason'] == 'stop' for r in replies), replies
assert all('4' in r['choices'][0]['message']['content'] for r in replies), replies
ids = post('/tokenize', {'messages': [{'role': 'user', 'content': 'Write a long explanation of TCP.'}]})['tokens']
req = urllib.request.Request(a.url+'/generate', data=json.dumps({'input_ids': ids, 'stream': True}).encode(),
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=180) as r:
    for line in r:
        if line.startswith(b'data:'):
            break
for _ in range(50):
    health = json.load(urllib.request.urlopen(a.url+'/health', timeout=30))
    if health['active_jobs'] == 0:
        break
    time.sleep(0.1)
assert health['active_jobs'] == 0, health
print(json.dumps({'passed': True, 'checks': ['invalid IDs', 'four concurrent natural completions',
                   'disconnect cancellation'], 'example': replies[0]}, indent=2))
