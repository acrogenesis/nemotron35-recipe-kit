"""Small ExLlamaV3 server implementing the recipe kit's streaming/quality API."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

os.environ.setdefault('EXL3_INT8_GEMV', '0')
os.environ.setdefault('EXL3_MOE_CPU_SWAP', '0')

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from exllamav3 import model_init, AsyncGenerator, AsyncJob
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.version import __version__
import uvicorn


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    model_init.add_args(ap, cache=True, default_cache_size=131072,
                        default_chunk_size=2048, default_autosplit_max_batch_size=4)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=30000)
    ap.add_argument('--model-name', default='nemotron35-exl3')
    ap.add_argument('--request-log', default='/results/requests.jsonl')
    return ap


def create_app(args):
    runtime = {}

    @asynccontextmanager
    async def lifespan(app):
        if args.chunk_size < 256 or args.chunk_size % 256:
            raise ValueError('chunk_size must be a positive multiple of 256')
        torch.set_num_threads(args.moe_cpu_threads)
        started = time.monotonic()
        model, config, cache, tokenizer = model_init.init(args)
        for host in getattr(config, 'moe_cpu_hosts', {}).values():
            host.ensure_started()
        hf = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
        gen = AsyncGenerator(model=model, cache=cache, tokenizer=tokenizer,
                             max_batch_size=args.autosplit_max_batch_size,
                             max_chunk_size=args.chunk_size, max_q_size=8,
                             recurrent_cache_size=int(args.recurrent_cache_size * 1024**3))
        stop = sorted(set(config.eos_token_id_list + [hf.eos_token_id]) - {None})
        vocab_ids = sorted(set(hf.get_vocab().values()))
        runtime.update(gen=gen, hf=hf, tokenizer=tokenizer, stop=stop,
                       vocab_size=config.vocab_size, vocab_ids=vocab_ids,
                       quality_lock=asyncio.Lock(), load_seconds=time.monotonic()-started)
        Path(args.request_log).parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({'ready': True, 'seconds': runtime['load_seconds'],
                          'max_batch_size': gen.generator.max_batch_size}), flush=True)
        try:
            yield
        finally:
            await gen.close()
            model.unload()
            config.stc.close()

    app = FastAPI(lifespan=lifespan)

    def write_log(row):
        with open(args.request_log, 'a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def render(body):
        if 'messages' in body:
            prompt = runtime['hf'].apply_chat_template(
                body['messages'], tokenize=False, add_generation_prompt=True,
                enable_thinking=body.get('chat_template_kwargs', {}).get('enable_thinking', False))
        else:
            prompt = body.get('prompt', body.get('text', ''))
        if not isinstance(prompt, str):
            raise HTTPException(400, 'prompt must be text')
        ids = runtime['tokenizer'].encode(prompt, encode_special_tokens=True, add_bos=False)[0].tolist()
        return prompt, ids

    def validate_ids(ids):
        if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < runtime['vocab_size'] for i in ids):
            raise HTTPException(400, 'input_ids must contain valid integer token IDs')
        if len(ids) >= args.cache_size:
            raise HTTPException(400, 'prompt fills the configured context')
        return torch.tensor([ids], dtype=torch.long)

    def require_greedy(body):
        if body.get('temperature', 0) != 0:
            raise HTTPException(400, 'This benchmark server supports temperature=0 only')

    @app.get('/health')
    async def health():
        if runtime['gen'].error is not None:
            raise HTTPException(503, str(runtime['gen'].error))
        return {'status': 'ok', 'active_jobs': len(runtime['gen'].jobs)}

    @app.get('/server_info')
    async def info():
        return {'model': args.model_name, 'engine': f'exllamav3 {__version__}',
                'context_length': args.cache_size, 'vocab_size': runtime['vocab_size'],
                'valid_random_token_ids': [i for i in runtime['vocab_ids'] if i >= 1000 and i not in runtime['hf'].all_special_ids],
                'max_running_requests': runtime['gen'].generator.max_batch_size,
                'max_total_num_tokens': args.cache_size, 'chunked_prefill_size': args.chunk_size,
                'kv_cache_dtype': 'float16', 'max_mamba_cache_size': args.autosplit_max_batch_size,
                'cpu_experts': args.moe_cpu_split, 'cpu_threads': args.moe_cpu_threads,
                'int8_gemv': os.environ['EXL3_INT8_GEMV'], 'dynamic_expert_swap': os.environ['EXL3_MOE_CPU_SWAP'],
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'load_seconds': runtime['load_seconds']}

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': args.model_name, 'object': 'model', 'owned_by': 'local'}]}

    @app.post('/tokenize')
    async def tokenize(body: dict):
        prompt, ids = render(body)
        return {'tokens': ids, 'prompt': prompt,
                'sha256': hashlib.sha256(prompt.encode()).hexdigest()}

    async def events(ids, limit, request):
        if runtime['quality_lock'].locked():
            raise HTTPException(409, 'Quality check in progress')
        job = AsyncJob(runtime['gen'], input_ids=validate_ids(ids), max_new_tokens=limit,
                       max_rq_tokens=2048,
                       sampler=ArgmaxSampler(), stop_conditions=runtime['stop'],
                       decode_special_tokens=False)
        started = time.perf_counter()
        count, text, final, first = 0, '', None, None
        try:
            async for event in job:
                if await request.is_disconnected():
                    break
                if event.get('stage') != 'streaming':
                    continue
                tokens = event.get('token_ids')
                count += tokens.numel() if tokens is not None else 0
                text += event.get('text', '')
                if not count and not event.get('eos'):
                    continue
                if first is None:
                    first = time.perf_counter() - started
                finish = None
                if event.get('eos'):
                    final = event
                    finish = {'type': 'length' if event['eos_reason'] == 'max_new_tokens' else 'stop',
                              'engine_reason': event['eos_reason']}
                yield {'text': text, 'delta': event.get('text', ''), 'meta_info': {
                    'prompt_tokens': len(ids), 'completion_tokens': event.get('new_tokens', count),
                    'finish_reason': finish, 'cached_tokens': event.get('cached_tokens')}}
        finally:
            await job.cancel()
            write_log({'time': time.time(), 'prompt_tokens': len(ids),
                       'prompt_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                       'completion_tokens': final['new_tokens'] if final else count,
                       'finish_reason': final['eos_reason'] if final else 'client_disconnected',
                       'cached_tokens': final.get('cached_tokens') if final else job.job.cached_tokens,
                       'seconds': time.perf_counter()-started, 'ttft_seconds': first, 'text': text})

    async def logprobs(body, ids):
        if runtime['quality_lock'].locked() or runtime['gen'].jobs:
            raise HTTPException(409, 'Quality checks require an idle server')
        start = int(body.get('logprob_start_len', 0))
        if not 0 <= start < len(ids):
            raise HTTPException(400, 'Invalid logprob_start_len')
        support = body.get('token_ids_logprob', [])
        if support:
            validate_ids(support)
        k = int(body.get('top_logprobs_num', 20))
        if not 1 <= k <= 100 or len(ids)-start > 4096:
            raise HTTPException(400, 'Quality request exceeds scoring limits')
        async with runtime['quality_lock']:
            p = start + 1
            job = AsyncJob(runtime['gen'], input_ids=validate_ids(ids[:p]),
                           max_new_tokens=len(ids)-p+1, sampler=ArgmaxSampler(),
                           stop_conditions=[], return_logits=True, decode_special_tokens=True)
            if p < len(ids):
                job.job.constrain_output_now(torch.tensor([ids[p:]], dtype=torch.long))
            tops, supports = [], []
            try:
                async for event in job:
                    logits = event.get('logits')
                    if logits is None:
                        continue
                    for row in logits.reshape(-1, logits.shape[-1]):
                        lp = row[:runtime['vocab_size']].float().log_softmax(-1)
                        values, indices = lp.topk(k)
                        tops.append([[float(v), int(i)] for v, i in zip(values.tolist(), indices.tolist())])
                        supports.append([[float(v), i] for v, i in zip(lp[support].tolist(), support)])
            finally:
                await job.cancel()
            assert len(tops) == len(ids)-start, (len(tops), len(ids)-start)
            return {'meta_info': {'input_top_logprobs': [[]]+tops[:-1],
                    'input_token_ids_logprobs': [[]]+supports[:-1],
                    'output_top_logprobs': [tops[-1]], 'output_token_ids_logprobs': [supports[-1]]}}

    @app.post('/generate')
    async def generate(body: dict, request: Request):
        ids = body.get('input_ids')
        if ids is None:
            _, ids = render(body)
        validate_ids(ids)
        sampling = body.get('sampling_params', {})
        require_greedy(sampling)
        limit = sampling.get('max_new_tokens')
        if limit is not None and (type(limit) is not int or limit < 1 or len(ids)+limit > args.cache_size):
            raise HTTPException(400, 'Invalid generation length for configured context')
        if body.get('return_logprob'):
            return await logprobs(body, ids)
        if body.get('stream'):
            async def stream():
                async for row in events(ids, limit, request):
                    yield 'data: '+json.dumps(row)+'\n\n'
                yield 'data: [DONE]\n\n'
            return StreamingResponse(stream(), media_type='text/event-stream')
        final = None
        async for final in events(ids, limit, request):
            pass
        return final

    @app.post('/v1/chat/completions')
    async def chat(body: dict, request: Request):
        require_greedy(body)
        _, ids = render(body)
        validate_ids(ids)
        limit = body.get('max_completion_tokens', body.get('max_tokens'))
        if limit is not None and (type(limit) is not int or limit < 1 or len(ids)+limit > args.cache_size):
            raise HTTPException(400, 'Invalid generation length')
        rid = 'chatcmpl-'+uuid.uuid4().hex
        created = int(time.time())
        def envelope(row, streaming):
            reason = row['meta_info']['finish_reason']
            choice = {'index': 0, 'finish_reason': reason['type'] if reason else None}
            choice['delta' if streaming else 'message'] = {'role': 'assistant', 'content': row['delta' if streaming else 'text']}
            return {'id': rid, 'object': 'chat.completion.chunk' if streaming else 'chat.completion',
                    'created': created, 'model': args.model_name, 'choices': [choice],
                    'usage': {'prompt_tokens': len(ids), 'completion_tokens': row['meta_info']['completion_tokens'],
                              'total_tokens': len(ids)+row['meta_info']['completion_tokens']}}
        if body.get('stream'):
            async def stream():
                async for row in events(ids, limit, request):
                    yield 'data: '+json.dumps(envelope(row, True))+'\n\n'
                yield 'data: [DONE]\n\n'
            return StreamingResponse(stream(), media_type='text/event-stream')
        final = None
        async for final in events(ids, limit, request):
            pass
        return envelope(final, False)

    return app


if __name__ == '__main__':
    args = parser().parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level='warning')
