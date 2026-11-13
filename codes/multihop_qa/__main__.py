"""python -m codes.multihop_qa --help"""
import argparse
import json
from pathlib import Path
from .schema import read_jsonl
from .pipeline import ingest_stage, extraction_stage, mining_stage, generation_stage, load
from .providers import JSONClient, LocalJSONClient, load_env_file
from .export import export_sft
from .experiment_cli import register, run as run_experiment, EXTRA_COMMANDS


def main():
    parser = argparse.ArgumentParser(description="Evidence-first multi-hop QA builder")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="parse documents and preserve evidence offsets")
    ingest.add_argument("input", type=Path)
    ingest.add_argument("--max-chars", type=int, default=1200)
    ingest.add_argument("--source-groups", type=Path, help="JSONL: source (path), source_group_id for known copies")
    ingest.add_argument('--pdf-backend',choices=['text','layout'],default='text')
    ingest.add_argument('--pages',help='inclusive physical page range, e.g. 1:8')
    ingest.add_argument('--ocr',choices=['auto','always','never'],default='auto')
    ingest.add_argument('--provisional-native',action='store_true',help='allow clean native text into generation; not human-reviewed')
    extraction = commands.add_parser("extract", help="load a grounded fact cache or explicitly call a model")
    extraction.add_argument("input", type=Path)
    extraction.add_argument("--facts-cache", type=Path)
    extraction.add_argument("--source-reviews", type=Path)
    extraction.add_argument("--aliases", type=Path)
    extraction.add_argument("--live", action="store_true")
    extraction.add_argument("--env-file", type=Path)
    extraction.add_argument("--model", default="qwen-plus")
    extraction.add_argument('--backend',choices=['api','local'],default='api')
    extraction.add_argument("--limit", type=int)
    mining = commands.add_parser("mine", help="build graph, recall candidates and rank evidence paths")
    mining.add_argument("input", type=Path)
    mining.add_argument("--embeddings", type=Path, help="JSONL: chunk_id, vector; all vectors from the same encoder")
    mining.add_argument("--beam-width", type=int, default=16)
    mining.add_argument("--path-limit", type=int, default=100)
    mining.add_argument("--recall-limit", type=int, default=32)
    mining.add_argument("--search", choices=["beam", "bfs"], default="beam")
    mining.add_argument("--scorer", type=Path)
    mining.add_argument('--implicit',action='store_true',help='bounded complementary-role recall without entity overlap; semantic verification still mandatory')
    generation = commands.add_parser("generate", help="local/API generation and exhaustive evidence verification")
    generation.add_argument("input", type=Path)
    generation.add_argument("--live", action="store_true")
    generation.add_argument("--env-file", type=Path)
    generation.add_argument("--generator-model", required=True)
    generation.add_argument("--verifier-model", required=True)
    generation.add_argument('--generator-backend',choices=['local','api'],default='local')
    generation.add_argument('--verifier-backend',choices=['local','api'],default='local')
    generation.add_argument('--device',default='cuda')
    generation.add_argument('--json-mode',choices=['schema','unconstrained'],default='schema',help='local generation syntax constraints; unconstrained is for diagnostics')
    generation.add_argument('--review-mode',choices=['independent','same-model-diagnostic'],default='independent')
    generation.add_argument("--threshold", type=float, default=.1)
    generation.add_argument('--calibration',type=Path,help='validated calibration.json; overrides threshold')
    generation.add_argument("--limit", type=int, default=10)
    export = commands.add_parser("export", help="export source-disjoint, human-approved SFT records")
    export.add_argument("input", type=Path)
    export.add_argument("--reviews", type=Path, required=True)
    demo = commands.add_parser("demo", help="offline synthetic integration example; not production QA")
    training = commands.add_parser("train-scorer", help="train a linear complement scorer from source-disjoint human labels")
    training.add_argument("input", type=Path)
    training.add_argument("--epochs", type=int, default=500)
    for command in (ingest, extraction, mining, generation, export, demo, training):
        command.add_argument("--output", required=True, type=Path)
    register(commands)
    args = parser.parse_args()
    try:
        if args.command in EXTRA_COMMANDS:
            result=run_experiment(args)
        elif args.command == "ingest":
            groups = {str(Path(r["source"]).resolve()): r["source_group_id"] for r in read_jsonl(args.source_groups)} if args.source_groups else None
            result = ingest_stage(args.input, args.output, args.max_chars, groups,args.pdf_backend,args.pages,args.ocr,args.provisional_native)
        elif args.command == "extract":
            if not args.facts_cache and not args.live:
                parser.error("use --facts-cache for offline extraction, or explicitly set --live")
            if args.limit is not None and args.limit < 1:
                parser.error("limit must be positive")
            if args.live and args.env_file:
                load_env_file(args.env_file)
            aliases = json.loads(args.aliases.read_text(encoding="utf8")) if args.aliases else None
            client=(LocalJSONClient(args.model) if args.backend=='local' else JSONClient(args.model)) if args.live else None
            result = extraction_stage(args.input, args.output, client,
                                      read_jsonl(args.facts_cache) if args.facts_cache else None,
                                      read_jsonl(args.source_reviews) if args.source_reviews else None,
                                      aliases, args.limit)
        elif args.command == "mine":
            vectors = {r["chunk_id"]: r["vector"] for r in read_jsonl(args.embeddings)} if args.embeddings else None
            scorer = json.loads(args.scorer.read_text(encoding="utf8")) if args.scorer else None
            result = mining_stage(args.input, args.output, vectors, args.beam_width, args.path_limit, args.recall_limit, args.search, scorer,args.implicit)
        elif args.command == "generate":
            if not args.live:
                parser.error("model execution requires --live")
            if args.env_file:
                load_env_file(args.env_file)
            calibration=json.loads(args.calibration.read_text(encoding='utf8')) if args.calibration else None
            if calibration and (calibration.get('calibrated') is not True or not calibration.get('validation_samples')):
                parser.error('invalid or uncalibrated threshold artifact')
            threshold=calibration['threshold'] if calibration else args.threshold
            def client(backend,model):
                return LocalJSONClient(model,device=args.device,json_mode=args.json_mode) if backend=='local' else JSONClient(model)
            generator=client(args.generator_backend,args.generator_model)
            verifier=client(args.verifier_backend,args.verifier_model)
            if generator.model==verifier.model and args.review_mode=='same-model-diagnostic':
                verifier=generator
            result = generation_stage(args.input, args.output, generator,
                                      verifier, threshold, args.limit,calibration,args.review_mode)
        elif args.command == "export":
            result = export_sft(load(args.input, "qa"), {c["chunk_id"]: c for c in load(args.input, "chunks")},
                                read_jsonl(args.reviews), args.output)
        elif args.command == "train-scorer":
            from .training import train_scorer
            result = train_scorer(read_jsonl(args.input), args.output, args.epochs)
        else:
            from .demo import run_demo
            result = run_demo(args.output)
    except (ValueError, OSError, KeyError, TypeError) as ex:
        parser.exit(1, f"{type(ex).__name__}: {ex}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("errors", 0) or result.get("rejected", 0) and args.command == "extract" else 0


if __name__ == "__main__":
    raise SystemExit(main())
