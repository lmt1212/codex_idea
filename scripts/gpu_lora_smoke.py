"""One synthetic optimizer step to validate GPU/LoRA plumbing, not an SFT experiment."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from codes.multihop_qa.schema import empty_output,write_json
from codes.multihop_qa.sft import model_preflight,encode_supervised


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    status=model_preflight(a.model)
    if not status['ready']:
        raise ValueError('model incomplete')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model
    if not torch.cuda.is_available():
        raise ValueError('GPU smoke test requires CUDA')
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    messages=[dict(role='user',content='这是合成测试。资料：甲使用工具乙；工具乙能够连接数据。甲使用的工具具有什么能力？'),
              dict(role='assistant',content='甲使用工具乙，工具乙能够连接数据。')]
    row=encode_supervised(messages,tokenizer,512)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.config.use_cache=False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    parameters=[param for param in model.parameters() if param.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=2e-4)
    loss=model(**{key:torch.tensor([value],device='cuda') for key,value in row.items()}).loss
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss')
    loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
    if not norm>0:
        raise ValueError('no trainable gradient')
    optimizer.step()
    output=empty_output(a.output)
    model.save_pretrained(output/'test_adapter')
    report=dict(status='gpu_lora_step_passed',synthetic_only=True,optimizer_steps=1,loss=float(loss.detach()),
                gradient_norm=norm,trainable_parameters=sum(p.numel() for p in parameters),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(0),
                note='Compute smoke test only, not a trained research model or a quality comparison.')
    write_json(output/'report.json',report)
    print(report)


if __name__=='__main__':
    main()
