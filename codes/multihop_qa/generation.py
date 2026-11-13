"""Generate only from a semantically checked path; persist independently reviewed QA."""
from .schema import VERSION, verification_fingerprint
from .verification import verify_path, validate_qa, verify_support, verify_necessity


def question_structure_issues(qa):
    """Reject a narrow pseudo-comparison before expensive model judging.

    A list of two independent values is a lookup task, even when it has two
    citations.  This check is deliberately limited to comparison paths; other
    task types still depend on support and necessity verification.
    """
    if qa['task_type'] != 'comparison':
        return []
    question = qa['question']
    issues = []
    if '分别' in question and any(marker in question for marker in ('什么', '哪些', '多少')):
        issues.append('enumerative_comparison_question')
    if not any(marker in question for marker in ('差异', '区别', '不同', '比较', '相比')):
        issues.append('comparison_question_lacks_difference_goal')
    if not any(len(claim['fact_ids']) >= 2 for claim in qa['answer_claims']):
        issues.append('comparison_answer_lacks_joint_claim')
    return issues


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
             requirements='比较类型须以差异、区别或相比为目标，不得问“分别是什么”；至少一个连接结论claim必须同时引用两个fact_ids，并完整保留各自数量或条件。每条路径事实均需参与回答，不得改为只用一处证据的定义题。'))
    if result.get("qa") is None:
        return None, dict(stage="generation", path_id=path["path_id"], reason="insufficient_evidence")
    qa = validate_qa(result["qa"], path, facts)
    structure_issues = question_structure_issues(qa)
    if structure_issues:
        return None, dict(stage='question_structure', qa=qa, issues=structure_issues)
    support = verify_support(qa, chunks, verifier)
    if not support["passed"]:
        return None, dict(stage="answer_support", qa=qa, check=support)
    necessity = verify_necessity(qa, chunks, verifier, threshold)
    qa.update(path_check=path_check, support_check=support, necessity_check=necessity,
              necessity_score=necessity.get("necessity_score"),
              generation_metadata=dict(pipeline_version=VERSION, prompt_version="2",
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
