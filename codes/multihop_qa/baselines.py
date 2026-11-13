"""Comparable evidence-selection baselines and explicitly unverified QA generation."""
import random
from collections import defaultdict
from .schema import uid, empty_output, write_json, write_jsonl
from .retrieval import VectorLSH

METHODS = ('single','adjacent','random','entity','dense','graph','direct_multihop')


def baseline_plans(chunks, method, facts=None, vectors=None, paths=None, limit=20):
    if method not in METHODS or limit < 1:
        raise ValueError('invalid baseline method/budget')
    by_doc, by_id = defaultdict(list), {c['chunk_id']:c for c in chunks}
    for c in chunks:
        if not c['needs_review'] and c.get('eligible_for_experiment',True):
            by_doc[c['doc_id']].append(c)
    eligible = [c for rows in by_doc.values() for c in rows]
    groups = []
    if method == 'single':
        groups = [[c['chunk_id']] for c in eligible]
    elif method == 'adjacent':
        original=defaultdict(list)
        for c in chunks:
            original[c['doc_id']].append(c)
        groups = [[a['chunk_id'],b['chunk_id']] for rows in original.values() for a,b in zip(rows,rows[1:])
                  if not a['needs_review'] and not b['needs_review'] and a.get('eligible_for_experiment',True)
                  and b.get('eligible_for_experiment',True) and a['section'] == b['section']]
    elif method == 'random':
        shuffled = list(eligible)
        random.Random(42).shuffle(shuffled)
        groups = [[a['chunk_id'],b['chunk_id']] for a,b in zip(shuffled[::2],shuffled[1::2])]
    elif method == 'direct_multihop':
        for i,seed in enumerate(eligible):
            others=[c for c in eligible if c['doc_id']!=seed['doc_id']]
            if not others:
                others=[c for c in eligible if c['chunk_id']!=seed['chunk_id']]
            if others:
                groups.append([seed['chunk_id'],others[i%len(others)]['chunk_id']])
    elif method == 'dense':
        if not vectors:
            raise ValueError('dense baseline requires real embeddings')
        index = VectorLSH(vectors)
        groups = [[c['chunk_id'],r[0]] for c in eligible for r in index.query(c['chunk_id'],1) if r[0] in by_id]
    elif method == 'entity':
        mapping = defaultdict(list)
        for f in facts or []:
            for key in {f['subject_id'],f['object_id']}:
                mapping[key].append(f['chunk_id'])
        groups = [list(dict.fromkeys(ids))[:2] for ids in mapping.values() if len(set(ids)) >= 2]
    else:
        groups = [p['evidence_chunk_ids'] for p in paths or []]
    seen, result = set(), []
    for ids in groups:
        if any(i not in by_id or by_id[i]['needs_review'] or not by_id[i].get('eligible_for_experiment',True) for i in ids):
            continue
        signature = tuple(sorted(ids))
        if signature in seen:
            continue
        seen.add(signature)
        result.append(dict(plan_id=uid('baseline',[method,ids]),method=method,evidence_chunk_ids=ids,
                           source_group_ids=sorted({by_id[i]['source_group_id'] for i in ids})))
        if len(result) >= limit:
            break
    return result


def run_baseline(chunks, output, method, client=None, facts=None, vectors=None, paths=None, limit=20):
    output = empty_output(output)
    plans = baseline_plans(chunks,method,facts,vectors,paths,limit)
    mapping = {c['chunk_id']:c for c in chunks}
    qas, errors = [], []
    for plan in plans if client else []:
        try:
            task_instruction=('请生成需要组合至少两处证据的多跳问题；不要把两个独立子题简单拼接。' if method=='direct_multihop' else '允许单跳。')
            result = client.ask('baseline_generation',
                '仅依据给定资料生成一道有实际价值的问题和答案，不补常识、不编造因果。'+task_instruction+
                '只要存在能直接回答的明确事实，就围绕该事实生成；不要求整篇文档完整，不要求多跳。'
                '只能返回有效JSON，所有键和字符串必须使用英文双引号。证据不足返回 {"qa":null}。'
                '否则严格返回 {"qa":{"question":"问题文本","answer":"答案文本"}}。',
                dict(evidence=[mapping[c] for c in plan['evidence_chunk_ids']]))
            item = result.get('qa')
            if item is None:
                continue
            if not isinstance(item,dict) or not all(isinstance(item.get(k),str) and item[k].strip() for k in ('question','answer')):
                raise ValueError('invalid baseline QA schema')
            qas.append(dict(qa_id=uid('baseline_qa',[plan['plan_id'],item]),**item,**plan,
                            verification_status='baseline_unverified',generator=client.model))
        except Exception as ex:
            errors.append(dict(plan_id=plan['plan_id'],error_type=type(ex).__name__,
                               reason=str(ex) if isinstance(ex,ValueError) else 'request_failed'))
        write_jsonl(output/'qa.jsonl',qas)
    for name,rows in [('plans',plans),('qa',qas),('errors',errors)]:
        write_jsonl(output/(name+'.jsonl'),rows)
    write_jsonl(output/'chunks.jsonl',chunks)
    report = dict(method=method,plans=len(plans),qa=len(qas),errors=len(errors),live=client is not None,
                  usage=client.history if client else [],note='Baseline outputs require identical blinded evaluation; not auto-approved SFT.')
    write_json(output/'report.json',report)
    return report
