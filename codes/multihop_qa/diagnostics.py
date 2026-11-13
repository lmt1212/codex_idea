"""Offline failure bundles and frozen input manifests; never create gold labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from .schema import read_jsonl,write_jsonl,write_json,empty_output


def diagnose_runs(folders, output):
    output=empty_output(output)
    summaries,failures,manifests=[],[],[]
    for folder in map(Path,folders):
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        def rows(name):
            p=folder/(name+'.jsonl')
            return read_jsonl(p) if p.exists() else []
        chunks={c['chunk_id']:c for c in rows('chunks')}
        facts={f['fact_id']:f for f in rows('facts')}
        paths={p['path_id']:p for p in rows('paths')}
        rejected,errors=rows('rejected'),rows('errors')
        calls=report.get('usage',[]) if report.get('stage')=='extraction' else report.get('generator_usage',[])+report.get('verifier_usage',[])
        summaries.append(dict(folder=str(folder),stage=report.get('stage'),
            attempted=report.get('attempted',report.get('attempted_chunks')),
            facts=report.get('facts'),auto_passed=report.get('auto_passed'),
            rejected=len(rejected),errors=len(errors),
            rejection_stages=dict(Counter(r.get('stage','unknown') for r in rejected)),
            calls=len(calls),calls_with_full_response=sum(c.get('raw_response') is not None for c in calls),
            calls_missing_timing=sum('elapsed_seconds' not in c for c in calls),
            elapsed_call_seconds=sum(c.get('elapsed_seconds',0) for c in calls),
            note='Missing historical fields remain unknown; elapsed excludes calls without timing.'))
        selected=[c['chunk_id'] for c in chunks.values() if not c['needs_review'] and c.get('eligible_for_experiment',True)]
        manifests.append(dict(folder=str(folder),eligible_chunk_ids=selected,
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.json*') if p.is_file()}))
        for kind,records in [('error',errors),('rejection',rejected)]:
            for record in records:
                qa=record.get('qa',{})
                path=paths.get(record.get('path_id',qa.get('path_id')), {})
                ids=set(qa.get('evidence_chunk_ids',path.get('evidence_chunk_ids',[])))
                if record.get('chunk_id'):ids.add(record['chunk_id'])
                failures.append(dict(folder=str(folder),kind=kind,record=record,path=path,
                    facts=[f for f in facts.values() if f['chunk_id'] in ids],
                    evidence=[chunks[cid] for cid in sorted(ids) if cid in chunks],
                    missing_evidence_ids=sorted(ids-set(chunks))))
    write_jsonl(output/'failures.jsonl',failures)
    write_jsonl(output/'input_manifests.jsonl',manifests)
    result=dict(runs=summaries,total_failure_records=len(failures),human_labels_created=0)
    write_json(output/'report.json',result)
    return result
