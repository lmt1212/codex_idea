"""Human-approved baseline datasets are experimental controls, not multihop claims."""
from .schema import empty_output,uid,write_json,write_jsonl
from .export import split_records


def release_baseline(qas,chunks,reviews,output):
    mapping={r['qa_id']:r for r in reviews}
    accepted,rejected=[],[]
    for qa in qas:
        r=mapping.get(qa['qa_id'],{})
        good=(r.get('human_reviewed') is True and r.get('decision')=='approved' and bool(r.get('reviewer'))
              and all(r.get(k) is True for k in ('answer_correct','evidence_faithful','question_clear')))
        if not good or qa.get('demo_only'):
            rejected.append(dict(qa_id=qa['qa_id'],reason='complete_independent_human_review_required'))
            continue
        expected=sorted({chunks[c]['source_group_id'] for c in qa['evidence_chunk_ids']})
        if expected!=sorted(qa['source_group_ids']):
            raise ValueError('baseline source metadata mismatch')
        accepted.append(qa)
    splits=split_records(accepted)
    output=empty_output(output)
    audit=[]
    for split in ('train','val','test'):
        rows=[]
        for qa in accepted:
            if splits[qa['qa_id']]!=split:
                continue
            context='\n\n'.join(f"[{c}] {chunks[c]['text']}" for c in qa['evidence_chunk_ids'])
            row=dict(messages=[dict(role='system',content='只根据给定资料回答，证据不足时说明不足。'),
                     dict(role='user',content=context+'\n问题：'+qa['question']),dict(role='assistant',content=qa['answer'])])
            rows.append(row)
            audit.append(dict(qa_id=qa['qa_id'],split=split,source_group_ids=qa['source_group_ids'],
                              review=mapping[qa['qa_id']],messages_fingerprint=uid('sft_messages',row),
                              dataset_purpose='baseline_experiment'))
        write_jsonl(output/(split+'.jsonl'),rows)
    write_jsonl(output/'audit.jsonl',audit)
    write_jsonl(output/'rejected.jsonl',rejected)
    report=dict(input=len(qas),exported=len(accepted),rejected=len(rejected),dataset_purpose='baseline_experiment')
    write_json(output/'report.json',report)
    return report
