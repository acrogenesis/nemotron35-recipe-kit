"""Build an independent EXL3 reference with full-GPU layer streaming, no CPU experts.

The fixed continuations are teacher forced, not reference greedy completions.
The speed server never reads this file; it returns its own per-position logits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('EXL3_INT8_GEMV', '0')
os.environ.setdefault('EXL3_MOE_CPU_SWAP', '0')
import torch
from transformers import AutoTokenizer
from exllamav3 import Config, Model, Tokenizer
from exllamav3.util.memory import free_mem, malloc_trim

CASES = [
    ('Explain how a hash table handles collisions.', 'A hash table maps keys to array positions using a hash function. When two keys map to the same position, the table can store them in a linked list or search for another empty slot. Resizing the array helps keep lookups efficient.'),
    ('Write a Python function to reverse a string.', '```python\ndef reverse_string(text: str) -> str:\n    return text[::-1]\n```\nPython slicing with a step of negative one visits the characters in reverse order. The original string remains unchanged.'),
    ('What is 17 multiplied by 23? Show the calculation.', 'We can split 23 into 20 and 3. First, 17 times 20 is 340. Then 17 times 3 is 51. Adding them gives 340 plus 51, which equals 391.'),
    ('Explica para qué sirve una base de datos.', 'Una base de datos permite guardar y organizar información para consultarla después. Por ejemplo, una tienda puede almacenar productos, precios y pedidos. Las consultas ayudan a encontrar los datos necesarios sin revisar todos los registros manualmente.'),
    ('How does TCP recover from packet loss?', 'TCP numbers the bytes in a stream and uses acknowledgments to track delivery. If data is lost, the sender retransmits it after a timeout or after receiving duplicate acknowledgments. Congestion control also adjusts the sending rate to reduce pressure on the network.'),
    ('Explain the difference between a process and a thread.', 'A process normally has its own memory space and operating system resources. Threads belong to a process and share its memory. This makes communication between threads convenient, but shared data needs synchronization to avoid races.'),
]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    config = Config.from_directory(a.model)
    tokenizer = Tokenizer.from_config(config)
    hf = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    panel, ids = [], []
    for question, continuation in CASES:
        prompt = hf.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer.encode(prompt, encode_special_tokens=True, add_bos=False)[0].tolist()
        tail = tokenizer.encode(continuation, encode_special_tokens=True, add_bos=False)[0].tolist()
        panel.append({'tokens': prefix+tail, 'prompt_len': len(prefix),
                      'question': question, 'continuation': continuation,
                      'reference': 'EXL3 weights, layer streamed on GPU; fixed teacher-forced continuation'})
        ids.append(torch.tensor([prefix+tail], dtype=torch.long))
    config.override_dynamic_seq_len(max(x.shape[-1] for x in ids))
    model = Model.from_config(config)
    params = [{'attn_mode': 'flash_attn_nc'} for _ in ids]
    states = [model.prepare_inputs(x, p) for x, p in zip(ids, params)]
    for index, module in enumerate(model.modules):
        start = time.monotonic()
        config.stc.begin_deferred_load()
        module.load(torch.device('cuda:0'))
        config.stc.end_deferred_load()
        for i, p in enumerate(params):
            states[i] = module.forward(module.prepare_for_device(states[i], p), p).cpu()
            assert torch.isfinite(states[i]).all(), (module.key, i)
        module.unload()
        config.stc.close()
        free_mem(); malloc_trim()
        print(json.dumps({'module': index, 'key': module.key, 'seconds': time.monotonic()-start}), flush=True)
    for item, logits in zip(panel, states):
        lp = logits[0, item['prompt_len']-1:, :config.vocab_size].float().log_softmax(-1)
        values, indices = lp.topk(20, dim=-1)
        item.update(top20_ids=indices.tolist(), top20_lp=values.tolist())
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(panel, indent=1)+'\n')
    print(json.dumps({'positions': sum(len(p['top20_ids']) for p in panel),
                      'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
