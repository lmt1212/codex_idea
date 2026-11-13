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


def formula_prefix_issues(values, source):
    """Reject an extracted formula prefix whose source has a short suffix.

    This is a deliberately narrow guard for layout line-wraps such as
    ``REx(C2O4)\ny``. It does not interpret chemical meaning or repair text.
    """
    compact = re.sub(r'\s+', '', source)
    source_formulas = re.findall(r'[A-Za-z][A-Za-z0-9()]*', compact)
    issues=[]
    for value in values:
        for candidate in re.findall(r'[A-Za-z][A-Za-z0-9()]*', re.sub(r'\s+', '', value)):
            if ')' not in candidate or not any(char.isdigit() for char in candidate):
                continue
            for full in source_formulas:
                suffix=full[len(candidate):]
                if full.startswith(candidate) and re.fullmatch(r'[xyz0-9]+',suffix):
                    issues.append(dict(extracted_fragment=candidate,source_fragment=full,
                                       reason='formula_suffix_omitted'))
    return issues


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
    formula_issues=formula_prefix_issues((f['subject'],f['predicate'],f['object']),f['quote'])
    if formula_issues:
        raise ValueError('extracted formula is truncated: '+formula_issues[0]['source_fragment'])
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


def definition_context(content, quote):
    """Recover only a contiguous heading already present in the same chunk."""
    start=content.index(quote)
    prefix=content[:start]
    lines=[m for m in re.finditer(r'[^\n]+',prefix) if m.group().strip()]
    if lines:
        last=lines[-1]
        if not re.search(r'[。！？；]',last.group()) and re.search(r'[\u4e00-\u9fff]',last.group()) and len(last.group())<=100:
            quote=content[last.start():start+len(quote)]
    hint=''
    # A bilingual glossary heading is an explicit term, not an inferred alias.
    for line in quote.splitlines()[:5]:
        match=re.fullmatch(r'\s*([\u4e00-\u9fff][\u4e00-\u9fff（）()·-]{1,39})\s+[A-Za-z][A-Za-z ,，-]*\s*',line)
        if match:
            hint=match[1]
            break
    return quote,hint


def extract_staged(chunk,client,aliases=None):
    selected=client.ask('select_source',
        '从text中选择表达明确事实的连续完整原文，逐字复制，保留换行、条件、否定和时间。定义句省略主体时必须连同前面的术语标题一起选取，不能只摘定义后半句。只输出JSON：{"quote":"逐字原句"}。不得改写或拼接不连续内容。',
        dict(text=chunk['text']))
    quote=selected.get('quote')
    if not isinstance(quote,str) or not quote or chunk['text'].count(quote)!=1:
        raise ValueError('selected source quote missing or ambiguous')
    quote,subject_hint=definition_context(chunk['text'],quote)
    triple=client.ask('extract_triple',
        '从给定原句抽取一个三元组，subject、predicate、object都必须是原句中的连续原文子串，不能改字或补常识。'
        'subject_hint非空时，主体必须使用该原文术语，不得用定义中的残句替代。'
        '只输出JSON：{"subject":"主体原文","predicate":"关系原文","object":"客体原文"}。',dict(text=quote,subject_hint=subject_hint))
    for key in ('subject','predicate','object'):
        if not isinstance(triple.get(key),str) or not triple[key] or triple[key] not in quote:
            raise ValueError('triple endpoint not anchored to selected quote: '+key)
    if subject_hint and triple['subject']!=subject_hint:
        raise ValueError('subject does not match explicitly anchored glossary term')
    if getattr(client,'require_fact_review',False) and re.match(r'^(?:以.+为原料|为一种|由.+组成的物质)',triple['subject']):
        raise ValueError('subject is a definition fragment rather than an explicit term')
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
    if getattr(client,'require_fact_review',False):
        check=client.ask('fact_completeness',
            '核验候选三元组是否完整表达原文事实。主体必须是明确命名的实体/术语，不能是“以…为原料”等残句；'
            '关系和客体不得改变原义，条件、数量、否定和不确定性不得遗漏。逐项判断，无法确认给false。'
            '返回严格JSON：{"subject_explicit":false,"relation_complete":false,"qualifiers_complete":false,"reason":"具体依据"}。',
            dict(source=chunk['text'],fact=raw))
        if not all(check.get(k) is True for k in ('subject_explicit','relation_complete','qualifiers_complete')):
            raise ValueError('fact completeness rejected: '+str(check.get('reason','unknown')))
    return [validate_fact(raw,chunk,aliases)]


def validate_cached_fact(fact, chunk, aliases=None):
    if chunk["needs_review"]:
        raise ValueError("source requires review")
    validated = validate_fact(fact, chunk, aliases)
    for key in ("fact_id", "subject_id", "object_id", "doc_id", "source_group_id"):
        if fact.get(key) != validated[key]:
            raise ValueError("stale or altered fact field: " + key)
    return validated
