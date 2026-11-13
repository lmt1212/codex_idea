"""Generate only from a semantically checked path; persist independently reviewed QA."""
from .schema import VERSION, verification_fingerprint
from .verification import verify_path, validate_qa, verify_support, verify_necessity


def generate_qa(path, facts, chunks, generator, verifier, threshold=.1, review_mode='independent'):
    diagnostic=review_mode=='same-model-diagnostic'
    if review_mode not in {'independent','same-model-diagnostic'}:
        raise ValueError('unknown review mode')
    if generator.model == verifier.model and not diagnostic:
        raise ValueError("generation and verification models must differ")
    path_check = verify_path(path, facts, chunks, verifier)
    if not path_check["passed"]:
        return None, dict(stage="path_verification", path_id=path["path_id"], check=path_check)
    result = generator.ask("generation",
        "根据核验后的证据结构生成一道需要多份证据组合的问题与答案。不要把多个独立问题拼接，"
        "不要泄漏桥接答案，不引入原文未支持的动机或因果。返回{qa:null}表示无法构造。否则返回"
        "{qa:{question,answer,answer_claims:[{text,fact_ids:[string]}]}}。每个text必须是答案中的连续"
        "唯一片段，所有实质答案文本都必须由不重叠的claims覆盖，包含连接结论。只能引用给定fact_ids。",
        dict(path=path, facts=[facts[f] for f in path["fact_ids"]],
             evidence=[chunks[c] for c in path["evidence_chunk_ids"]],
             requirements='比较类型必须询问两个对象的共同维度；每条路径事实均需参与回答，不得改为只用一处证据的定义题。'))
    if result.get("qa") is None:
        return None, dict(stage="generation", path_id=path["path_id"], reason="insufficient_evidence")
    qa = validate_qa(result["qa"], path, facts)
    support = verify_support(qa, chunks, verifier)
    if not support["passed"]:
        return None, dict(stage="answer_support", qa=qa, check=support)
    necessity = verify_necessity(qa, chunks, verifier, threshold)
    qa.update(path_check=path_check, support_check=support, necessity_check=necessity,
              necessity_score=necessity.get("necessity_score"),
              generation_metadata=dict(pipeline_version=VERSION, prompt_version="1",
                                       generator=generator.model, verifier=verifier.model),
              verification_status="auto_passed_pending_human_review" if necessity["passed"] else "rejected")
    if not necessity["passed"]:
        return None, dict(stage="necessity", qa=qa)
    # Returned identities, not just configured aliases, must differ too.
    generated = {h["actual_model"] for h in generator.history}
    judged = {h["actual_model"] for h in verifier.history}
    if generated & judged and not diagnostic:
        raise ValueError("generator and verifier resolved to the same actual model")
    qa['independent_model_review']=not bool(generated & judged) and generator.model!=verifier.model
    if diagnostic:
        qa['verification_status']='diagnostic_pending_independent_and_human_review'
    qa["verification_fingerprint"] = verification_fingerprint(qa, chunks)
    return qa, None
