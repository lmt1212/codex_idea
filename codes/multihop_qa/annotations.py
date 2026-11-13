"""Blinded, provenance-rich annotation packs; responses always start unlabelled."""
from pathlib import Path
from .schema import uid, empty_output, write_jsonl, write_json, read_jsonl

QA_FIELDS = ('answer_correct','evidence_faithful','true_multihop','reasoning_valid','question_clear')


def create_pack(runs, output, limit=100, kind='qa'):
    if kind not in {'qa','pair'} or limit < 1:
        raise ValueError('invalid annotation pack settings')
    output = empty_output(output)
    tasks, responses, mapping, seen = [], [], [], set()
    for spec in runs:
        folder = Path(spec['folder'])
        chunks = {c['chunk_id']:c for c in read_jsonl(folder/'chunks.jsonl')}
        rows = read_jsonl(folder/('qa.jsonl' if kind == 'qa' else 'candidates.jsonl'))
        if kind=='qa' and spec.get('include_rejected') and (folder/'rejected.jsonl').exists():
            rows += [r['qa'] for r in read_jsonl(folder/'rejected.jsonl') if isinstance(r.get('qa'),dict)]
        facts = {f['fact_id']:f for f in read_jsonl(folder/'facts.jsonl')} if kind == 'pair' else {}
        for row in rows[:limit]:
            if kind == 'qa':
                record_id = row['qa_id']
                ids = row['evidence_chunk_ids']
                display = dict(question=row['question'],answer=row['answer'])
            else:
                record_id = uid('pair',sorted([row['source'],row['target']]))
                ids = sorted({facts[f]['chunk_id'] for f in (row['source'],row['target'])})
                display = dict(facts=[facts[row['source']],facts[row['target']]])
            if record_id in seen:
                continue
            seen.add(record_id)
            aid = uid('annotation',[record_id,kind])
            groups = sorted({chunks[c]['source_group_id'] for c in ids})
            tasks.append(dict(annotation_id=aid,kind=kind,**display,
                              evidence=[chunks[c] for c in ids],
                              content_fingerprint=uid('annotation_content',[display,[chunks[c] for c in ids]])))
            response = dict(annotation_id=aid,human_reviewed=False,reviewer='',decision=None,
                            reason='',split=spec.get('split','val'))
            if kind == 'qa':
                response.update({k:None for k in QA_FIELDS})
                response['hop_count'] = None
            else:
                response['label'] = None
            responses.append(response)
            mapping.append(dict(annotation_id=aid,record_id=record_id,method=spec['method'],kind=kind,
                                source_group_ids=groups,source_folder=str(folder),
                                features=row.get('scores'),fact_ids=[row['source'],row['target']] if kind == 'pair' else None,
                                necessity_score=row.get('necessity_score')))
    # Interleave methods using stable blinded IDs, no method labels in public tasks.
    tasks.sort(key=lambda r:r['annotation_id'])
    responses.sort(key=lambda r:r['annotation_id'])
    write_jsonl(output/'tasks.jsonl',tasks)
    write_jsonl(output/'responses.jsonl',responses)
    write_jsonl(output/'private_mapping.jsonl',mapping)
    report = dict(kind=kind,items=len(tasks),human_labelled=0,status='awaiting_human_annotation',
                  instructions='Review original pages/conditions. Do not show private_mapping to blinded reviewers. Never auto-fill labels.')
    write_json(output/'report.json',report)
    (output/'标注说明.md').write_text('# 人工盲评说明\n\n逐条查看 tasks.jsonl 的问题、答案与证据，必要时按页码核对原书。\n'
        '填写 responses.jsonl；没有把握的项目保持 null，不能猜测。完成后填写真实 reviewer，设置 human_reviewed=true。\n'
        'decision 为 approved/rejected/review；只有完整核验后才能批准。QA 的 hop_count 按必要事实依赖填写，不能按引用块数填写。\n'
        'pair 的 label 为 1（逻辑互补）或0（不互补/重复/冲突），并说明判断。\n'
        'private_mapping.jsonl 仅供实验负责人汇总，不提供给盲评标注者。\n',encoding='utf8')
    return report


def completed_reviews(pack):
    folder = Path(pack)
    mapping = {r['annotation_id']:r for r in read_jsonl(folder/'private_mapping.jsonl')}
    responses = read_jsonl(folder/'responses.jsonl')
    result, seen = [], set()
    for r in responses:
        aid = r['annotation_id']
        if aid in seen or aid not in mapping:
            raise ValueError('duplicate/unknown annotation response')
        seen.add(aid)
        if r.get('human_reviewed') is not True:
            continue
        if not isinstance(r.get('reviewer'),str) or not r['reviewer'].strip():
            raise ValueError('reviewer required')
        if r.get('decision') not in {'approved','rejected','review'}:
            raise ValueError('invalid decision')
        meta = mapping[aid]
        if meta['kind'] == 'qa':
            if any(type(r.get(k)) is not bool for k in QA_FIELDS):
                raise ValueError('complete boolean QA labels required')
            if type(r.get('hop_count')) is not int or r['hop_count'] < 0:
                raise ValueError('human-verified hop count required')
            result.append(dict(r,qa_id=meta['record_id'],**meta))
        else:
            if type(r.get('label')) is not int or r['label'] not in {0,1}:
                raise ValueError('pair label must be integer 0 or 1')
            result.append(dict(r,**meta))
    return result, len(mapping)


def source_review_pack(chunks,output,limit=30):
    output=empty_output(output)
    selected=sorted(chunks,key=lambda c:(not c['needs_review'],c['doc_id'],c['chunk_id']))[:limit]
    write_jsonl(output/'source_tasks.jsonl',selected)
    write_jsonl(output/'source_reviews.jsonl',[dict(chunk_id=c['chunk_id'],decision=None,human_reviewed=False,
                                                  reviewer='',reason='') for c in selected])
    report=dict(items=len(selected),status='awaiting_original_page_review',human_reviewed=0,
                instructions='Compare text/table/formula to the indicated original PDF page. Do not approve from OCR confidence alone.')
    write_json(output/'report.json',report)
    return report
