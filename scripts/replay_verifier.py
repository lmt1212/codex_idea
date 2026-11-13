"""Replay an immutable recorded request across local models and decoding modes."""
import argparse
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from codes.multihop_qa.schema import read_jsonl,empty_output,write_json,write_jsonl
from codes.multihop_qa.providers import LocalJSONClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--models',nargs='+',required=True)
    parser.add_argument('--modes',nargs='+',choices=['schema','unconstrained'],default=['unconstrained','schema'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records=read_jsonl(args.run/'verifier_calls.jsonl')
    failed=[r for r in records if r.get('status')=='failed']
    if not failed:raise ValueError('no frozen failed verifier request found')
    frozen=failed[0]
    output=empty_output(args.output)
    write_json(output/'frozen_request.json',frozen)
    results,calls=[],[]
    for model in args.models:
        client=LocalJSONClient(model,max_calls=len(args.modes))
        for mode in args.modes:
            client.json_mode=mode
            row=dict(model=model,json_mode=mode,task=frozen['task'],status='failed',diagnostic_only=True)
            try:
                row['response']=client.ask(frozen['task'],frozen['instruction'],frozen['input'],_retry_of=-1)
                row['status']='parsed'
            except Exception as ex:
                row.update(error_type=type(ex).__name__,error=str(ex))
            results.append(row)
            write_jsonl(output/'results.jsonl',results)
        calls.extend(client.history)
        write_jsonl(output/'calls.jsonl',calls)
        del client
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    write_json(output/'report.json',dict(attempts=len(results),parsed=sum(r['status']=='parsed' for r in results),
        results=results,independent_validation=False,human_labels_created=0,
        limitation='One frozen request; format conformance is not semantic accuracy. Replaying the generator as judge is diagnostic only.'))


if __name__=='__main__':main()
