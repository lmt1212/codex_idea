"""GPU LoRA SFT with answer-only loss and source-audited experiment manifests."""
import importlib.util
import json
import math
from pathlib import Path
from .schema import uid, read_jsonl, write_json, write_jsonl, empty_output


def model_preflight(model_path):
    path = Path(model_path)
    missing = []
    if not (path/'config.json').exists():
        missing.append('config.json')
    index_file = path/'model.safetensors.index.json'
    if index_file.exists():
        index = json.loads(index_file.read_text())
        expected = sorted(set(index['weight_map'].values()))
    else:
        expected = ['model.safetensors']
    for name in expected:
        if not (path/name).is_file() or (path/name).stat().st_size == 0:
            missing.append(name)
    if not any((path/name).exists() for name in ('tokenizer.json','tokenizer.model')):
        missing.append('tokenizer.json or tokenizer.model')
    return dict(model_path=str(path),ready=not missing,missing=missing,
                incomplete_files=[p.name for p in path.glob('*.incomplete')],
                dependencies={name:bool(importlib.util.find_spec(name)) for name in ('torch','transformers','peft')})


def encode_supervised(messages, tokenizer, max_tokens):
    if not isinstance(messages,list) or len(messages)<2 or messages[-1].get('role')!='assistant':
        raise ValueError('SFT requires a final assistant answer')
    full = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
    ids = tokenizer.encode(full,add_special_tokens=False)
    prefix = tokenizer.encode(prompt,add_special_tokens=False)
    if ids[:len(prefix)] != prefix:
        raise ValueError('chat template does not preserve prompt token prefix')
    if len(ids)>max_tokens:
        raise ValueError('SFT record exceeds token budget; do not silently truncate evidence/answers')
    labels = [-100]*len(prefix)+ids[len(prefix):]
    if not any(v!=-100 for v in labels):
        raise ValueError('no assistant tokens to train')
    return dict(input_ids=ids,attention_mask=[1]*len(ids),labels=labels)


def validate_release(folder):
    folder = Path(folder)
    audit = read_jsonl(folder/'audit.jsonl')
    group_splits = {}
    for item in audit:
        review = item.get('review',{})
        if review.get('human_reviewed') is not True or review.get('decision')!='approved' or not review.get('reviewer'):
            raise ValueError('SFT requires human-approved release audit')
        for group in item['source_group_ids']:
            if group in group_splits and group_splits[group]!=item['split']:
                raise ValueError('SFT source-group leakage')
            group_splits[group]=item['split']
    splits = {s:read_jsonl(folder/(s+'.jsonl')) for s in ('train','val','test')}
    for name,rows in splits.items():
        if sum(r['split']==name for r in audit)!=len(rows):
            raise ValueError('SFT audit and split row counts differ')
        for row,record in zip(rows,[r for r in audit if r['split']==name]):
            if record.get('messages_fingerprint')!=uid('sft_messages',row):
                raise ValueError('SFT messages differ from audited content')
    if not splits['train'] or not splits['val']:
        raise ValueError('nonempty source-disjoint train/val required for SFT')
    return splits,audit


def train_lora(model_path,dataset,output,max_steps=100,max_tokens=2048,accumulation=4,learning_rate=2e-4,seed=42):
    if min(max_steps,max_tokens,accumulation)<1:
        raise ValueError('positive SFT budgets required')
    preflight = model_preflight(model_path)
    if not preflight['ready'] or not all(preflight['dependencies'].values()):
        raise ValueError('SFT environment/model not ready: '+json.dumps(preflight))
    splits,audit = validate_release(dataset)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    train = [encode_supervised(r['messages'],tokenizer,max_tokens) for r in splits['train']]
    val = [encode_supervised(r['messages'],tokenizer,max_tokens) for r in splits['val']]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=dtype,attn_implementation='sdpa').to(device)
    model = get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
                     target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    def tensor(record):
        return {k:torch.tensor([v],device=device) for k,v in record.items()}
    def evaluate():
        model.eval()
        total, count=0.,0
        with torch.no_grad():
            for row in val:
                tokens=sum(v!=-100 for v in row['labels'])
                total+=float(model(**tensor(row)).loss)*tokens
                count+=tokens
        model.train()
        return total/count if count else None
    output=empty_output(output)
    before=evaluate()
    logs,assistant_tokens=[],0
    optimizer.zero_grad(set_to_none=True)
    for step in range(max_steps):
        loss_sum=0.
        for micro in range(accumulation):
            row=train[(step*accumulation+micro)%len(train)]
            loss=model(**tensor(row)).loss
            if not torch.isfinite(loss):
                raise ValueError('nonfinite SFT loss')
            (loss/accumulation).backward()
            loss_sum+=float(loss.detach())
            assistant_tokens+=sum(v!=-100 for v in row['labels'])
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logs.append(dict(step=step+1,train_loss=loss_sum/accumulation,assistant_tokens=assistant_tokens))
        write_jsonl(output/'train_log.jsonl',logs)
    after=evaluate()
    model.save_pretrained(output/'adapter')
    tokenizer.save_pretrained(output/'adapter')
    report=dict(model=str(model_path),dataset=str(dataset),max_steps=max_steps,train_samples=len(train),val_samples=len(val),
                assistant_tokens=assistant_tokens,max_tokens=max_tokens,accumulation=accumulation,seed=seed,
                learning_rate=learning_rate,val_loss_before=before,val_loss_after=after,
                device=device,max_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                note='Loss is not multi-hop answer accuracy; evaluate on an independent benchmark.')
    write_json(output/'report.json',report)
    return report


def prepare_experiment(arms,model_path,output,max_steps=100,max_tokens=2048):
    if set(arms)!={'single','direct_multihop','graph','full'}:
        raise ValueError('four arms required: single/direct_multihop/graph/full')
    datasets={}
    audits={}
    ownership={}
    for arm,folder in arms.items():
        splits,audit=validate_release(folder)
        datasets[arm]=splits
        audits[arm]=audit
        for row in audit:
            for group in row['source_group_ids']:
                if group in ownership and ownership[group]!=row['split']:
                    raise ValueError('cross-arm train/evaluation source leakage')
                ownership[group]=row['split']
    count=min(len(v['train']) for v in datasets.values())
    output=empty_output(output)
    prepared_arms={}
    manifest=dict(base_model=str(model_path),max_steps=max_steps,max_tokens=max_tokens,seed=42,
                  equal_train_samples=count,original_arms=arms,model_preflight=model_preflight(model_path),
                  status='ready_for_training' if model_preflight(model_path)['ready'] else 'awaiting_complete_model',
                  note='Shared steps/sample count; record actual assistant tokens. Do not claim all token budgets equal.')
    for arm,data in datasets.items():
        destination=output/arm
        destination.mkdir()
        training_audit=[r for r in audits[arm] if r['split']=='train']
        pairs=sorted(zip(data['train'],training_audit),key=lambda pair:uid('sample',pair[0]))[:count]
        selected=[p[0] for p in pairs]
        audit_selected=[p[1] for p in pairs]+[r for r in audits[arm] if r['split']!='train']
        for split in ('train','val','test'):
            write_jsonl(destination/(split+'.jsonl'),selected if split=='train' else data[split])
        write_jsonl(destination/'audit.jsonl',audit_selected)
        prepared_arms[arm]=str(destination)
    manifest['arms']=prepared_arms
    manifest['commands']={arm:f'python -m codes.multihop_qa sft-train {folder} --model {model_path} --steps {max_steps} --max-tokens {max_tokens} --output TRAIN_OUTPUT_{arm}' for arm,folder in prepared_arms.items()}
    write_json(output/'experiment.json',manifest)
    return manifest
