"""Fact extraction, role labels, qualifiers and exact evidence coordinates."""
import re
import json
from .schema import uid, text, strings, ROLES

PROMPT = ('从输入text中抽取1到3条明确陈述的事实，不生成问答。必须返回英文双引号的JSON对象，根键为facts，值为数组。'
          '每条事实按下列格式填写，quote逐字复制完整原句，保留条件、否定、时间，不补外部知识。'
          'role只允许'+','.join(sorted(ROLES))+'。time无明确时间时为空字符串，event无事件时为空字符串。'
          '结构示例：'+json.dumps(dict(facts=[dict(subject='实体',predicate='关系',object='对象',role='attribute',quote='逐字原句',conditions=[],time='',polarity='affirmed',modality='certain',event='')]),ensure_ascii=False)
          +'。确实无可抽取事实时返回 {"facts":[]}，不能返回空对象。')


def canonical(name, aliases):
    normalized = re.sub(r"\s+", "", name).casefold()
    return aliases.get(normalized, normalized)


def validate_fact(raw, chunk, aliases=None):
    if not isinstance(raw, dict):
        raise ValueError("fact must be an object")
    f = {k: text(raw.get(k), k) for k in ("subject", "predicate", "object", "quote", "role")}
    if f["role"] not in ROLES:
        raise ValueError("unsupported fact role")
    for field in ("polarity", "modality"):
        f[field] = raw.get(field)
    if f["polarity"] not in {"affirmed", "negated"} or f["modality"] not in {"certain", "uncertain"}:
        raise ValueError("missing or invalid polarity/modality")
    f["conditions"] = strings(raw.get("conditions"), "conditions")
    if not isinstance(raw.get("time"), str) or not isinstance(raw.get("event", ""), str):
        raise ValueError("time and event must be strings")
    f.update(time=raw["time"], event=raw.get("event", ""))
    start = chunk["text"].find(f["quote"])
    if start < 0 or chunk["text"].find(f["quote"], start + 1) >= 0:
        raise ValueError("quote missing or ambiguous")
    if "start" in raw and (raw["start"] != start or raw.get("end") != start + len(f["quote"])):
        raise ValueError("cached coordinates do not match source")
    if "chunk_id" in raw and raw["chunk_id"] != chunk["chunk_id"]:
        raise ValueError("cached chunk does not match")
    aliases = aliases or {}
    f.update(subject_id=canonical(f["subject"], aliases), object_id=canonical(f["object"], aliases),
             chunk_id=chunk["chunk_id"], doc_id=chunk["doc_id"],
             source_group_id=chunk["source_group_id"], start=start, end=start + len(f["quote"]))
    f["fact_id"] = uid("fact", f)
    f["status"] = "quote_validated_pending_semantic_review"
    return f


def extract(chunk, client, aliases=None):
    if chunk["needs_review"]:
        raise ValueError("source requires explicit layout/OCR review")
    if getattr(client,'extraction_strategy','structured')=='staged':
        return extract_staged(chunk,client,aliases)
    result = client.ask("extraction", PROMPT,dict(chunk_id=chunk['chunk_id'],text=chunk['text']))
    rows = result.get("facts")
    if not isinstance(rows, list) or len(rows) > 12:
        raise ValueError("invalid extraction schema")
    return [validate_fact(f, chunk, aliases) for f in rows]


def extract_staged(chunk,client,aliases=None):
    selected=client.ask('select_source',
        '从text中选择一个表达明确事实的完整原句，逐字复制，保留换行、条件、否定和时间。只输出JSON：{"quote":"逐字原句"}。不得改写。',
        dict(text=chunk['text']))
    quote=selected.get('quote')
    if not isinstance(quote,str) or not quote or chunk['text'].count(quote)!=1:
        raise ValueError('selected source quote missing or ambiguous')
    triple=client.ask('extract_triple',
        '从给定原句抽取一个三元组，subject、predicate、object都必须是原句中的连续原文子串，不能改字或补常识。'
        '只输出JSON：{"subject":"主体原文","predicate":"关系原文","object":"客体原文"}。',dict(text=quote))
    for key in ('subject','predicate','object'):
        if not isinstance(triple.get(key),str) or not triple[key] or triple[key] not in quote:
            raise ValueError('triple endpoint not anchored to selected quote: '+key)
    qualifiers=client.ask('classify_fact',
        '判断三元组在原句中的角色及限定条件，不修改三元组。role只允许'+','.join(sorted(ROLES))+'. '
        'conditions只填写原句中逐字存在的条件片段，无则[]。time只填原文时间，无则空字符串。'
        'polarity为affirmed或negated，modality为certain或uncertain。严格JSON示例：'
        '{"role":"attribute","conditions":[],"time":"","polarity":"affirmed","modality":"certain","event":""}。',
        dict(text=quote,triple=triple))
    if not isinstance(qualifiers.get('conditions'),list) or any(not isinstance(c,str) or c not in quote for c in qualifiers['conditions']):
        raise ValueError('unanchored condition')
    if qualifiers.get('time') and qualifiers['time'] not in quote:
        raise ValueError('unanchored time')
    raw={key:triple[key] for key in ('subject','predicate','object')}
    raw.update({key:qualifiers.get(key) for key in ('role','conditions','time','polarity','modality')})
    raw.update(quote=quote,event=qualifiers.get('event',''))
    return [validate_fact(raw,chunk,aliases)]


def validate_cached_fact(fact, chunk, aliases=None):
    if chunk["needs_review"]:
        raise ValueError("source requires review")
    validated = validate_fact(fact, chunk, aliases)
    for key in ("fact_id", "subject_id", "object_id", "doc_id", "source_group_id"):
        if fact.get(key) != validated[key]:
            raise ValueError("stale or altered fact field: " + key)
    return validated
