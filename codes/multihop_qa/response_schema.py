"""Generation constraints describe syntax, never the correct evidence judgment."""


def obj(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def response_schema(task,data):
    string=dict(type='string')
    boolean=dict(type='boolean')
    if task=='path_verification':
        ids=[f['fact_id'] for f in data['facts']]
        row=obj(dict(fact_id=dict(type='string',enum=ids),supported=boolean,qualifiers_complete=boolean))
        return obj(dict(facts=dict(type='array',items=row,minItems=len(ids),maxItems=len(ids)),
                        entity_identity=boolean,relations_valid=boolean,common_target=boolean,reason=string))
    if task=='answer_support':
        ids=[c['claim_id'] for c in data['qa']['answer_claims']]
        row=obj(dict(claim_id=dict(type='string',enum=ids),entailed=boolean,qualifiers_complete=boolean,
                     citations_correct=boolean,missing_qualifiers=dict(type='array',items=string),assessment_reason=string))
        return obj(dict(claims=dict(type='array',items=row,minItems=len(ids),maxItems=len(ids)),
                        question_clear=boolean,answer_complete=boolean,unique_answer=boolean,connected_reasoning=boolean,reason=string))
    if task=='generation':
        claim=obj(dict(text=string,fact_ids=dict(type='array',items=dict(type='string',enum=data['path']['fact_ids']),minItems=1)))
        qa=obj(dict(question=string,answer=string,answer_claims=dict(type='array',items=claim,minItems=1)))
        return obj(dict(qa=dict(anyOf=[dict(type='null'),qa])))
    if task=='select_source':return obj(dict(quote=string))
    if task=='extract_triple':return obj(dict(subject=string,predicate=string,object=string))
    if task=='classify_fact':
        from .schema import ROLES
        return obj(dict(role=dict(type='string',enum=sorted(ROLES)),conditions=dict(type='array',items=string),
                        time=string,polarity=dict(type='string',enum=['affirmed','negated']),
                        modality=dict(type='string',enum=['certain','uncertain']),event=string))
    if task=='fact_completeness':
        return obj(dict(subject_explicit=boolean,relation_complete=boolean,qualifiers_complete=boolean,reason=string))
    if task=='blind_solve':
        return dict(type='object',properties=dict(answer=dict(type='string'),sufficient=dict(type='boolean')),
                    required=['answer','sufficient'],additionalProperties=False)
    if task=='answer_coverage':
        ids=[c['claim_id'] for c in data['target_claims']]
        refs=[c['citation_id'] for c in data['allowed_citations']]
        if not ids or not refs:
            raise ValueError('coverage schema requires claims and current evidence references')
        row=dict(type='object',properties=dict(claim_id=dict(type='string',enum=ids),
            covered=dict(type='boolean'),supported=dict(type='boolean'),
            citations=dict(type='array',items=dict(type='string',enum=refs))),
            required=['claim_id','covered','supported','citations'],additionalProperties=False)
        return dict(type='object',properties=dict(claims=dict(type='array',items=row,minItems=len(ids),maxItems=len(ids))),
                    required=['claims'],additionalProperties=False)
    # LMFE's documented schemaless JSON mode; an empty object schema is not
    # interchangeable with it for arbitrary nested objects.
    return None
