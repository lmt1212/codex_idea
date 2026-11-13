"""CLI registration for real-data stages, annotation and training experiments."""
import json
from pathlib import Path
from .schema import read_jsonl, empty_output, write_jsonl, write_json
from .pipeline import load
from .providers import JSONClient,LocalJSONClient,load_env_file

EXTRA_COMMANDS={'embed','baseline','annotate','quality','calibrate','sft-check','sft-train','sft-plan','benchmark','rechunk','release-baseline'}


def register(commands):
    embed=commands.add_parser('embed',help='generate/cache real API or local embeddings')
    embed.add_argument('input',type=Path)
    embed.add_argument('--backend',choices=['api','local'],default='local')
    embed.add_argument('--model',required=True)
    embed.add_argument('--device',default='cpu')
    embed.add_argument('--pooling',choices=['cls','mean'],default='cls')
    embed.add_argument('--batch-size',type=int,default=8)
    embed.add_argument('--cache',type=Path)
    embed.add_argument('--env-file',type=Path)
    embed.add_argument('--live',action='store_true')
    baseline=commands.add_parser('baseline',help='repeatable single/adjacent and retrieval baselines')
    baseline.add_argument('input',type=Path)
    baseline.add_argument('--method',choices=['single','adjacent','random','entity','dense','graph','direct_multihop'],required=True)
    baseline.add_argument('--limit',type=int,default=20)
    baseline.add_argument('--vectors',type=Path)
    baseline.add_argument('--live',action='store_true')
    baseline.add_argument('--model',default='qwen-plus')
    baseline.add_argument('--backend',choices=['api','local'],default='api')
    baseline.add_argument('--env-file',type=Path)
    release=commands.add_parser('release-baseline',help='export strictly human-reviewed baseline controls')
    release.add_argument('input',type=Path)
    release.add_argument('--reviews',type=Path,required=True)
    annotate=commands.add_parser('annotate',help='create blinded unlabelled evidence/QA annotation tasks')
    annotate.add_argument('input',type=Path,help='JSON list of {folder,method,split}')
    annotate.add_argument('--kind',choices=['qa','pair'],default='qa')
    annotate.add_argument('--limit',type=int,default=100)
    quality=commands.add_parser('quality',help='report only completed independent human labels')
    quality.add_argument('input',type=Path)
    calibration=commands.add_parser('calibrate',help='calibrate thresholds on human validation labels only')
    calibration.add_argument('input',type=Path)
    calibration.add_argument('--min-precision',type=float,default=.9)
    check=commands.add_parser('sft-check',help='check complete model weights and environment')
    check.add_argument('input',type=Path)
    train=commands.add_parser('sft-train',help='real GPU LoRA training from audited train/val release')
    train.add_argument('input',type=Path)
    train.add_argument('--model',required=True,type=Path)
    train.add_argument('--steps',type=int,default=100)
    train.add_argument('--max-tokens',type=int,default=2048)
    train.add_argument('--accumulation',type=int,default=4)
    plan=commands.add_parser('sft-plan',help='prepare source-disjoint four-arm SFT experiment')
    plan.add_argument('input',type=Path,help='JSON object of single/direct_multihop/graph/full dataset folders')
    plan.add_argument('--model',required=True,type=Path)
    plan.add_argument('--steps',type=int,default=100)
    plan.add_argument('--max-tokens',type=int,default=2048)
    benchmark=commands.add_parser('benchmark',help='evaluate base/adapter on a human-reference holdout')
    benchmark.add_argument('input',type=Path)
    benchmark.add_argument('--model',type=Path,required=True)
    benchmark.add_argument('--adapter',type=Path)
    benchmark.add_argument('--training-release',type=Path)
    rechunk=commands.add_parser('rechunk',help='token-budget-aware exact-offset evidence splitting')
    rechunk.add_argument('input',type=Path)
    rechunk.add_argument('--tokenizer',required=True)
    rechunk.add_argument('--max-tokens',type=int,default=480)
    for command in (embed,baseline,release,annotate,quality,calibration,check,train,plan,benchmark,rechunk):
        command.add_argument('--output',required=True,type=Path)


def run(args):
    if args.command=='release-baseline':
        from .baseline_release import release_baseline
        return release_baseline(load(args.input,'qa'),{c['chunk_id']:c for c in load(args.input,'chunks')},read_jsonl(args.reviews),args.output)
    if args.command=='embed':
        from .embeddings import embed_chunks
        if args.backend=='api' and not args.live:
            raise ValueError('embedding API requires --live')
        if args.env_file and args.live:
            load_env_file(args.env_file)
        return embed_chunks(load(args.input,'chunks'),args.output,args.model,args.backend,args.batch_size,args.cache,args.device,args.pooling)
    if args.command=='baseline':
        from .baselines import run_baseline
        if args.live and args.env_file:
            load_env_file(args.env_file)
        def optional(name):
            return load(args.input,name) if (args.input/(name+'.jsonl')).exists() else None
        vectors={r['chunk_id']:r['vector'] for r in read_jsonl(args.vectors)} if args.vectors else None
        client=(LocalJSONClient(args.model,max_calls=args.limit) if args.backend=='local' else JSONClient(args.model,max_calls=args.limit)) if args.live else None
        return run_baseline(load(args.input,'chunks'),args.output,args.method,client,
                            optional('facts'),vectors,optional('paths'),args.limit)
    if args.command=='annotate':
        from .annotations import create_pack
        return create_pack(json.loads(args.input.read_text(encoding='utf8')),args.output,args.limit,args.kind)
    if args.command=='quality':
        from .quality import quality_report
        return quality_report(args.input,args.output)
    if args.command=='calibrate':
        from .calibration import calibrate
        return calibrate(read_jsonl(args.input),args.output,args.min_precision)
    if args.command=='sft-check':
        from .sft import model_preflight
        report=model_preflight(args.input)
        write_json(empty_output(args.output)/'report.json',report)
        return report
    if args.command=='sft-train':
        from .sft import train_lora
        return train_lora(args.model,args.input,args.output,args.steps,args.max_tokens,args.accumulation)
    if args.command=='sft-plan':
        from .sft import prepare_experiment
        return prepare_experiment(json.loads(args.input.read_text(encoding='utf8')),args.model,args.output,args.steps,args.max_tokens)
    if args.command=='benchmark':
        from .benchmark import run_benchmark
        return run_benchmark(args.model,read_jsonl(args.input),args.output,args.adapter,training_release=args.training_release)
    if args.command=='rechunk':
        from transformers import AutoTokenizer
        from .ingestion import token_rechunk
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=Path(args.tokenizer).exists())
        docs=load(args.input,'documents')
        chunks=token_rechunk(docs,tokenizer,args.max_tokens)
        output=empty_output(args.output)
        write_jsonl(output/'documents.jsonl',docs)
        write_jsonl(output/'chunks.jsonl',chunks)
        report=dict(chunks=len(chunks),max_tokens=args.max_tokens,tokenizer=args.tokenizer)
        write_json(output/'report.json',report)
        return report
    raise ValueError('unknown experiment command')
