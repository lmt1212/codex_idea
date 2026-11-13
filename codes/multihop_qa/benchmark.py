"""Independent human-authored held-out answer benchmark for base/adapter comparison."""
import re
from collections import Counter
from .schema import empty_output, write_jsonl, write_json


def normalize_answer(answer):
    return ''.join(c.casefold() for c in answer if c.isalnum())


def answer_metrics(prediction, references):
    pred = normalize_answer(prediction)
    scores=[]
    for reference in references:
        gold=normalize_answer(reference)
        common=sum((Counter(pred)&Counter(gold)).values())
        precision=common/len(pred) if pred else 0.
        recall=common/len(gold) if gold else 0.
        scores.append(dict(exact_match=float(pred==gold),
                           character_f1=2*precision*recall/(precision+recall) if precision+recall else 0.))
    return dict(exact_match=max(s['exact_match'] for s in scores),character_f1=max(s['character_f1'] for s in scores))


def run_benchmark(model_path, rows, output, adapter=None, max_new_tokens=256, training_release=None):
    if not rows:
        raise ValueError('nonempty independent benchmark required')
    for row in rows:
        if row.get('human_reviewed') is not True or not row.get('reviewer') or not row.get('references'):
            raise ValueError('benchmark requires human-verified reference answers')
    if adapter and not training_release:
        raise ValueError('adapter evaluation requires training-release audit to check source leakage')
    if training_release:
        from .schema import read_jsonl
        from pathlib import Path
        audit=read_jsonl(Path(training_release)/'audit.jsonl')
        forbidden={g for r in audit if r['split'] in {'train','val'} for g in r['source_group_ids']}
        for row in rows:
            if not row.get('source_group_ids'):
                raise ValueError('benchmark source groups required')
            if forbidden.intersection(row['source_group_ids']):
                raise ValueError('benchmark overlaps train/validation sources')
    from .sft import model_preflight
    if not model_preflight(model_path)['ready']:
        raise ValueError('base model incomplete')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter)
    model.eval()
    output=empty_output(output)
    results=[]
    for row in rows:
        context='\n\n'.join(row['evidence'])
        messages=[dict(role='system',content='只根据资料回答，证据不足时说明不足。'),
                  dict(role='user',content=context+'\n问题：'+row['question'])]
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(device)
        with torch.no_grad():
            ids=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        answer=tokenizer.decode(ids[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        results.append(dict(item_id=row['item_id'],prediction=answer,**answer_metrics(answer,row['references'])))
        write_jsonl(output/'predictions.jsonl',results)
    report=dict(items=len(results),model=str(model_path),adapter=str(adapter) if adapter else None,
                exact_match=sum(r['exact_match'] for r in results)/len(results),
                character_f1=sum(r['character_f1'] for r in results)/len(results),
                note='Surface answer metrics only; hallucination and reasoning validity require human evidence review.')
    write_json(output/'report.json',report)
    return report
