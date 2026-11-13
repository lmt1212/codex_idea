"""Fail-closed claim coverage, group ablations and single-evidence shortcut checks."""
import itertools
import re
from .schema import uid, text, strings, probability


def exact_rows(result, key, expected, id_key):
    rows = result.get(key)
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("invalid verification rows: " + key)
    ids = [r.get(id_key) for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("incomplete or duplicated verification IDs: " + key)
    return rows


def verify_path(path, facts, chunks, client):
    selected = [facts[fid] for fid in path["fact_ids"]]
    result = client.ask("path_verification",
        "核验候选路径每个事实的主体、关系、角色、条件、否定和时间是否被原文支持；"
        "核验显式或隐含桥接关系：同名不一定同指、不同名可能指同一概念；必须用原文说明逻辑联系。"
        "没有共享实体的路径也要逐条核验，不得只因角色互补就认可。核验关系兼容且足以组成共同推理目标。不能将采用技术推断为实际动机，"
        "比较任务允许且通常要求比较主体不同；entity_identity判断共享或桥接名称是否无歧义，不能因两个比较对象不同而给false。没有待消歧的共享名称时该项为true。"
        "不能将先后/目的改写为因果。返回{facts:[{fact_id,supported:bool,qualifiers_complete:bool}],"
        "entity_identity:bool,relations_valid:bool,common_target:bool,reason:string}。不能确认给false。",
        dict(path=path, facts=selected, evidence=[chunks[cid] for cid in path["evidence_chunk_ids"]]))
    rows = exact_rows(result, "facts", path["fact_ids"], "fact_id")
    result["passed"] = (all(r.get("supported") is True and r.get("qualifiers_complete") is True for r in rows)
                        and all(result.get(k) is True for k in ("entity_identity", "relations_valid", "common_target")))
    return result


def validate_qa(raw, path, facts):
    if not isinstance(raw, dict):
        raise ValueError("QA must be an object")
    question, answer = text(raw.get("question"), "question"), text(raw.get("answer"), "answer")
    rows = raw.get("answer_claims")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty answer_claims required")
    claims, covered_chars, covered_facts = [], set(), set()
    for raw_claim in rows:
        if not isinstance(raw_claim, dict):
            raise ValueError("invalid answer claim")
        value = text(raw_claim.get("text"), "claim text")
        start = answer.find(value)
        if start < 0 or answer.find(value, start + 1) >= 0:
            raise ValueError("claim must uniquely match answer text")
        ids = strings(raw_claim.get("fact_ids"), "claim fact IDs", nonempty=True)
        if not set(ids) <= set(path["fact_ids"]):
            raise ValueError("claim cites facts outside path")
        covered_facts.update(ids)
        span = set(range(start, start + len(value)))
        if covered_chars & span:
            raise ValueError("overlapping answer claims")
        covered_chars.update(span)
        claims.append(dict(claim_id=uid("claim", [value, ids]), text=value, start=start, end=start+len(value),
                           fact_ids=ids, evidence_chunk_ids=sorted({facts[f]["chunk_id"] for f in ids})))
    meaningful = {i for i, c in enumerate(answer) if c.isalnum()}
    if not meaningful <= covered_chars:
        raise ValueError("answer contains text outside cited claims")
    if covered_facts != set(path["fact_ids"]):
        raise ValueError("not all path facts contribute to the answer")
    if len({c["claim_id"] for c in claims}) != len(claims):
        raise ValueError("duplicate answer claims")
    return dict(qa_id=uid("qa", [path["path_id"], question, answer]), path_id=path["path_id"],
                question=question, answer=answer, answer_claims=claims, task_type=path["task_type"],
                evidence_chunk_ids=path["evidence_chunk_ids"], fact_ids=path["fact_ids"],
                source_group_ids=path["source_group_ids"], hop_count=None,
                candidate_steps=path["candidate_steps"], verification_status="pending")


def formula_truncations(qa,chunks):
    """Flag a narrow, explicit formula-prefix omission; not a chemistry parser."""
    issues=[]
    for claim in qa['answer_claims']:
        if '化学式' not in claim.get('text',''):
            continue
        claimed=re.findall(r'[A-Za-z][A-Za-z0-9()]*',re.sub(r'\s+','',claim['text']))
        for cid in claim.get('evidence_chunk_ids',[]):
            source=chunks[cid]['text']
            if '化学式' not in source:
                continue
            formulas=re.findall(r'[A-Za-z][A-Za-z0-9()]*',re.sub(r'\s+','',source))
            for value in claimed:
                if not any(c.isdigit() for c in value) or ')' not in value:
                    continue
                for full in formulas:
                    if full.startswith(value) and re.fullmatch(r'[xyz0-9]+',full[len(value):]):
                        issues.append(dict(claim_id=claim['claim_id'],chunk_id=cid,answer_fragment=value,
                                           source_fragment=full,reason='formula_suffix_omitted'))
    return issues


def verify_support(qa, chunks, client):
    issues=formula_truncations(qa,chunks)
    if issues:
        return dict(passed=False,deterministic_issues=issues,reason='quoted formula suffix missing from answer')
    result = client.ask("answer_support",
        "逐条检查答案事实是否被其引用证据支持，包括事实之间的推导。检查无遗漏条件、时间、否定、"
        "单位与范围；引用存在不代表支持。检查题意、答案完整性、共同推理目标和唯一性，排除并列拼题。"
        "每条claim额外返回missing_qualifiers字符串数组，明确指出影响本题结论却遗漏的条件，"
        "以及assessment_reason说明证据依据。qualifiers_complete为false时必须列出缺失项；不能确认时写无法确认的具体项。"
        "返回{claims:[{claim_id,entailed:bool,qualifiers_complete:bool,citations_correct:bool}],"
        "question_clear:bool,answer_complete:bool,unique_answer:bool,connected_reasoning:bool,reason:string}。",
        dict(qa=qa, evidence=[chunks[c] for c in qa["evidence_chunk_ids"]]))
    rows = exact_rows(result, "claims", [c["claim_id"] for c in qa["answer_claims"]], "claim_id")
    result['diagnostic_issues']=[]
    for row in rows:
        missing=row.get('missing_qualifiers')
        if row.get('qualifiers_complete') is False and not missing:
            result['diagnostic_issues'].append(dict(claim_id=row['claim_id'],reason='qualifier_rejection_without_specific_missing_item'))
        if row.get('qualifiers_complete') is True and missing:
            row['qualifiers_complete']=False
            result['diagnostic_issues'].append(dict(claim_id=row['claim_id'],reason='qualifier_decision_conflicts_with_missing_items'))
    result["passed"] = (all(all(r.get(k) is True for k in ("entailed", "qualifiers_complete", "citations_correct")) for r in rows)
                        and all(result.get(k) is True for k in ("question_clear", "answer_complete", "unique_answer", "connected_reasoning")))
    return result


def evidence_groups(qa, chunks):
    # Exact copies are alternatives, not independently necessary sources.
    # Near duplicates/semantic substitutes require supplied evidence_group_id.
    groups = {}
    for cid in qa["evidence_chunk_ids"]:
        chunk = chunks[cid]
        key = chunk.get("evidence_group_id") or uid("duplicate", re.sub(r"\s+", "", chunk["text"]))
        groups.setdefault(key, []).append(cid)
    return list(groups.values())


def verify_necessity(qa, chunks, client, threshold=.1):
    probability(threshold, "necessity threshold")
    ids = qa["evidence_chunk_ids"]
    groups = evidence_groups(qa, chunks)
    if not 2 <= len(groups) <= 4:
        return dict(passed=False, reason="requires 2-4 distinct evidence groups")
    claim_ids = [c["claim_id"] for c in qa["answer_claims"]]
    trials = []
    # At most 2^4 subsets. Exhaustive grouped trials include group deletion,
    # combinations, single groups and the empty-evidence control.
    selections = [(list(indexes), [cid for i in indexes for cid in groups[i]])
                  for length in range(len(groups) + 1)
                  for indexes in itertools.combinations(range(len(groups)), length)]
    existing = {tuple(sorted(selected)) for _, selected in selections}
    for cid in ids:
        if (cid,) not in existing:
            selections.append(([], [cid]))
    for indexes, selected in selections:
            selected_set = set(selected)
            catalogue={f'e{i}':dict(chunk_id=cid,quote=chunks[cid]['text']) for i,cid in enumerate(selected)}
            solved = client.ask("blind_solve",
                "只根据给定证据回答问题，不使用记忆或猜测。返回{answer:string,sufficient:bool}；"
                "缺少证据必须sufficient=false，不补全缺失事实。",
                dict(question=qa["question"], evidence=[chunks[c] for c in selected]))
            if not isinstance(solved.get("answer"), str) or type(solved.get("sufficient")) is not bool:
                raise ValueError("invalid blind solver schema")
            # With no evidence, supported coverage is exactly zero. Do not ask
            # a judge shown the target answer to invent evidence for this control.
            coverage = dict(claims=[dict(claim_id=cid,covered=False,supported=False,citations=[]) for cid in claim_ids]) if not selected else client.ask("answer_coverage",
                "对照目标答案事实，评估盲答是否覆盖且被当前证据支持。目标事实仅为评测标准，不能作为证据。"
                "每条返回claim_id、covered:bool、supported:bool和citations:[string]，只能选择allowed_citations中的citation_id。"
                "禁止生成或改写引用原句，禁止引用目标答案作为证据。"
                "缺证据、只复述题干或无依据推断都给false。返回{claims:[...]}。",
                dict(question=qa["question"], blind_answer=solved["answer"],
                     target_claims=[dict(claim_id=c['claim_id'],text=c['text']) for c in qa['answer_claims']],
                     allowed_citations=[dict(citation_id=k,**v) for k,v in catalogue.items()],
                     evidence=[chunks[c] for c in selected]))
            rows = exact_rows(coverage, "claims", claim_ids, "claim_id")
            supported = []
            for row in rows:
                valid = row.get("covered") is True and row.get("supported") is True
                citations = row.get("citations")
                if not isinstance(citations, list):
                    raise ValueError("coverage citations must be a list")
                if valid and not citations:
                    raise ValueError("supported claim lacks citations")
                resolved=[]
                for citation in citations:
                    if isinstance(citation,str):
                        if citation not in catalogue:
                            raise ValueError('coverage cites unknown catalogue reference')
                        citation=dict(catalogue[citation])
                    if not isinstance(citation, dict):
                        raise ValueError("invalid coverage citation")
                    cid = citation.get("chunk_id")
                    quote = text(citation.get("quote"), "coverage quote")
                    if cid not in selected_set or quote not in chunks[cid]["text"]:
                        raise ValueError("coverage cites absent evidence")
                    resolved.append(citation)
                row['citations']=resolved
                if valid:
                    supported.append(row["claim_id"])
            score = len(supported) / len(claim_ids)
            trials.append(dict(group_indexes=list(indexes), evidence_chunk_ids=selected,
                               score=score, sufficient=solved["sufficient"],
                               blind_answer=solved["answer"], claims=rows))
    full = next(t for t in trials if len(t["group_indexes"]) == len(groups))
    proper = [t for t in trials if len(t["group_indexes"]) < len(groups)]
    deletions = [t for t in proper if len(t["group_indexes"]) == len(groups)-1]
    marginal = [full["score"] - t["score"] for t in deletions]
    passed = (full["score"] == 1 and full["sufficient"] is True
              and all(t["score"] < 1 for t in proper)
              and min(marginal) > threshold)
    return dict(passed=passed, groups=groups, necessity_score=min(marginal), threshold=threshold,
                full_score=full["score"], trials=trials,
                method="exhaustive_evidence_group_ablation_with_claim_coverage",
                limitation="model-judged; human calibration and semantic substitute grouping required")
