"""Deterministic fixtures exercise plumbing only, never estimate real QA quality."""
from .schema import empty_output, write_jsonl, write_json
from .ingestion import document_from_text, chunk_documents
from .extraction import validate_fact
from .pipeline import extraction_stage, mining_stage, generation_stage, load
from .export import export_sft


class FixtureClient:
    """Not a model: deliberately simple, deterministic integration-test responses."""
    def __init__(self, model):
        self.model = model
        self.history = []

    def ask(self, task, instruction, data):
        self.history.append(dict(task=task, actual_model=self.model, usage=None, fixture_only=True))
        if task == "path_verification":
            return dict(facts=[dict(fact_id=f["fact_id"], supported=True, qualifiers_complete=True) for f in data["facts"]],
                        entity_identity=True, relations_valid=True, common_target=True)
        if task == "generation":
            facts = data["facts"]
            return dict(qa=dict(question="根据关联证据，企业采用的技术、其能力与相关问题之间有什么联系？",
                                answer="".join(f["quote"] for f in facts),
                                answer_claims=[dict(text=f["quote"], fact_ids=[f["fact_id"]]) for f in facts]))
        if task == "answer_support":
            return dict(claims=[dict(claim_id=c["claim_id"], entailed=True, qualifiers_complete=True, citations_correct=True)
                                for c in data["qa"]["answer_claims"]],
                        question_clear=True, answer_complete=True, unique_answer=True, connected_reasoning=True)
        if task == "blind_solve":
            return dict(answer="".join(c["text"] for c in data["evidence"]), sufficient=bool(data["evidence"]))
        if task == "answer_coverage":
            rows = []
            for claim in data["target_claims"]:
                citations = [dict(chunk_id=c["chunk_id"], quote=claim["text"]) for c in data["evidence"] if claim["text"] in c["text"]]
                covered = bool(citations) and claim["text"] in data["blind_answer"]
                rows.append(dict(claim_id=claim["claim_id"], covered=covered, supported=covered, citations=citations))
            return dict(claims=rows)
        raise ValueError("unsupported fixture task")


def fixture_documents():
    return [document_from_text("# 企业技术\n\n企业甲采用知识图谱。", "企业技术"),
            document_from_text("# 技术能力\n\n知识图谱能够整合异构数据。", "技术能力"),
            document_from_text("# 业务问题\n\n企业甲面临数据孤岛问题。", "业务问题")]


def fixture_facts(chunks):
    specifications = [("企业甲", "采用", "知识图谱", "adoption"),
                      ("知识图谱", "能够整合", "异构数据", "capability"),
                      ("企业甲", "面临", "数据孤岛", "problem")]
    return [dict(chunk_id=c["chunk_id"], subject=s, predicate=p, object=o, role=r, quote=c["text"],
                 conditions=[], time="", event="", polarity="affirmed", modality="certain")
            for c, (s, p, o, r) in zip(chunks, specifications)]


def run_demo(output):
    output = empty_output(output)
    prepared = output / "01_prepared"
    prepared.mkdir()
    docs = fixture_documents()
    chunks = chunk_documents(docs)
    write_jsonl(prepared / "documents.jsonl", docs)
    write_jsonl(prepared / "chunks.jsonl", chunks)
    extraction_stage(prepared, output / "02_extracted", cached_facts=fixture_facts(chunks))
    mining = mining_stage(output / "02_extracted", output / "03_mined")
    generated = generation_stage(output / "03_mined", output / "04_qa", FixtureClient("fixture-generator"), FixtureClient("fixture-verifier"))
    qas = load(output / "04_qa", "qa")
    for qa in qas:
        qa["demo_only"] = True
    write_jsonl(output / "04_qa" / "qa.jsonl", qas)
    exported = export_sft(qas, {c["chunk_id"]: c for c in chunks}, [], output / "05_release_gate")
    report = dict(mode="synthetic_offline_fixture", candidate_paths=mining["paths"],
                  fixture_auto_passed=generated["auto_passed"], errors=generated["errors"],
                  production_exported=exported["exported"],
                  warning="Integration plumbing only. Fixture checks do not validate reasoning or real model quality.")
    write_json(output / "demo_report.json", report)
    return report
