"""Generate real embeddings with model/text-bound resumable cache entries."""
import json
import os
import urllib.request
from pathlib import Path
from .schema import uid, empty_output, write_jsonl, write_json, read_jsonl
from .retrieval import normalize_vector


def embed_chunks(chunks, output, model='text-embedding-v3', backend='api', batch_size=8, cache=None, device='cpu', pooling='cls'):
    if batch_size < 1 or backend not in {'api','local'} or pooling not in {'cls','mean'}:
        raise ValueError('invalid embedding settings')
    output = empty_output(output)
    cached = {r['cache_key']:r for r in read_jsonl(cache)} if cache and Path(cache).exists() else {}
    vectors, pending, usage, skipped = [], [], [], []
    model_key = str(Path(model).resolve()) if backend == 'local' else model
    for chunk in chunks:
        if chunk.get('oversized'):
            skipped.append(dict(chunk_id=chunk['chunk_id'],reason='atomic_table_exceeds_token_budget'))
            continue
        key = uid('embedding', [backend, model_key, pooling, chunk['text']])
        if key in cached:
            record = dict(cached[key], chunk_id=chunk['chunk_id'])
            normalize_vector(record['vector'])
            vectors.append(record)
        else:
            pending.append((chunk,key))
    tokenizer = encoder = None
    if pending and backend == 'local':
        import torch
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=Path(model).exists())
        encoder = AutoModel.from_pretrained(model, local_files_only=Path(model).exists()).to(device).eval()
    for start in range(0,len(pending),batch_size):
        batch = pending[start:start+batch_size]
        texts = [c['text'] for c,_ in batch]
        if backend == 'api':
            from .schema import require_external_provider_allowed
            require_external_provider_allowed()
            key = os.environ.get('DASHSCOPE_API_KEY')
            if not key:
                raise ValueError('DASHSCOPE_API_KEY required for embedding API')
            base = os.environ.get('DASHSCOPE_BASE_URL','https://dashscope.aliyuncs.com/compatible-mode/v1')
            request = urllib.request.Request(base.rstrip('/')+'/embeddings',
                data=json.dumps(dict(model=model,input=texts,encoding_format='float')).encode(),
                headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            with urllib.request.urlopen(request,timeout=120) as response:
                result = json.load(response)
            data = sorted(result['data'],key=lambda r:r['index'])
            if [r['index'] for r in data] != list(range(len(batch))):
                raise ValueError('embedding response indexes incomplete')
            values = [r['embedding'] for r in data]
            usage.append(result.get('usage'))
        else:
            import torch
            lengths = [len(tokenizer.encode(t,add_special_tokens=True)) for t in texts]
            maximum = min(getattr(tokenizer,'model_max_length',512), getattr(encoder.config,'max_position_embeddings',512))
            if any(n > maximum for n in lengths):
                raise ValueError('embedding input exceeds encoder context; token-rechunk first')
            inputs = tokenizer(texts,padding=True,truncation=False,return_tensors='pt').to(device)
            with torch.no_grad():
                hidden = encoder(**inputs).last_hidden_state
                if pooling == 'cls':
                    features = hidden[:,0]
                else:
                    mask = inputs['attention_mask'].unsqueeze(-1)
                    features = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                values = features.float().cpu().tolist()
        if len(values) != len(batch):
            raise ValueError('embedding batch size mismatch')
        for (chunk,key), vector in zip(batch,values):
            record = dict(chunk_id=chunk['chunk_id'],cache_key=key,text_fingerprint=uid('text',chunk['text']),
                          model=model_key,backend=backend,pooling=pooling,vector=normalize_vector(vector))
            vectors.append(record)
            cached[key] = record
        write_jsonl(output/'vectors.jsonl',vectors)
        if cache:
            Path(cache).parent.mkdir(parents=True,exist_ok=True)
            write_jsonl(cache,cached.values())
    if len({len(r['vector']) for r in vectors}) > 1:
        raise ValueError('mixed embedding dimensions')
    write_jsonl(output/'vectors.jsonl',vectors)
    report = dict(chunks=len(chunks),vectors=len(vectors),model=model_key,backend=backend,
                  generated=len(pending),cache_hits=len(chunks)-len(pending)-len(skipped),skipped=skipped,usage=usage)
    write_json(output/'report.json',report)
    return report
