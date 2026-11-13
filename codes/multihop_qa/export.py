"""Release reviewed QA using the same IDs and records as the generation pipeline."""
from .schema import empty_output, uid, write_jsonl, write_json, verification_fingerprint


def split_records(records):
    """Connected source groups always stay together, including cross-doc samples."""
    parent = {}
    def root(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = root(parent[key])
        return parent[key]
    for row in records:
        groups = row["source_group_ids"]
        if not groups:
            raise ValueError("missing source group")
        for group in groups:
            parent[root(group)] = root(groups[0])
    components = {}
    for group in parent:
        components.setdefault(root(group), []).append(group)
    split_for = {}
    for group, members in components.items():
        bucket = int(uid("split", sorted(members)).split(":")[1][:8], 16) % 100
        split_for[group] = "train" if bucket < 80 else "val" if bucket < 90 else "test"
    return {row["qa_id"]: split_for[root(row["source_group_ids"][0])] for row in records}


def export_sft(qas, chunks, reviews, output):
    accepted, rejected, seen = [], [], set()
    review_map = {}
    for review in reviews:
        if review["qa_id"] in review_map:
            raise ValueError("duplicate human review")
        review_map[review["qa_id"]] = review
    for qa in qas:
        reason = None
        review = review_map.get(qa["qa_id"], {})
        if qa.get("demo_only") is True:
            reason = "synthetic_fixture_not_releasable"
        elif qa.get("verification_fingerprint") != verification_fingerprint(qa, chunks):
            reason = "content_changed_since_verification"
        elif qa["verification_status"] != "auto_passed_pending_human_review":
            reason = "not_auto_passed"
        elif not all(qa.get(k, {}).get("passed") is True for k in ("path_check", "support_check", "necessity_check")):
            reason = "verification_missing"
        elif not (review.get("decision") == "approved" and review.get("human_reviewed") is True
                  and isinstance(review.get("reviewer"), str) and review["reviewer"].strip()):
            reason = "human_review_required"
        source_groups = sorted({chunks[c]["source_group_id"] for c in qa["evidence_chunk_ids"]})
        if source_groups != sorted(qa["source_group_ids"]):
            raise ValueError("QA source groups differ from current evidence")
        signature = uid("dedup", ["".join(qa["question"].split()), "".join(qa["answer"].split()), source_groups])
        if signature in seen:
            reason = "duplicate_question_answer_scope"
        if reason:
            rejected.append(dict(qa_id=qa["qa_id"], reason=reason))
            continue
        seen.add(signature)
        accepted.append(qa)
    splits = split_records(accepted)
    output = empty_output(output)
    audits = []
    for split in ("train", "val", "test"):
        rows = []
        for qa in accepted:
            if splits[qa["qa_id"]] != split:
                continue
            context = "\n\n".join(f"[{cid}] {chunks[cid]['text']}" for cid in qa["evidence_chunk_ids"])
            rows.append(dict(messages=[dict(role="system", content="只根据给定资料回答，证据不足时说明不足。"),
                                       dict(role="user", content=context + "\n\n问题：" + qa["question"]),
                                       dict(role="assistant", content=qa["answer"])]))
            audits.append(dict(qa_id=qa["qa_id"], split=split, source_group_ids=qa["source_group_ids"],
                               messages_fingerprint=uid('sft_messages',rows[-1]),
                               evidence=[chunks[c] for c in qa["evidence_chunk_ids"]], review=review_map[qa["qa_id"]]))
        write_jsonl(output / (split + ".jsonl"), rows)
    write_jsonl(output / "audit.jsonl", audits)
    write_jsonl(output / "rejected.jsonl", rejected)
    report = dict(input=len(qas), exported=len(accepted), rejected=len(rejected),
                  split_counts={s: sum(v == s for v in splits.values()) for s in ("train", "val", "test")},
                  source_components=len({tuple(sorted(a["source_group_ids"])) for a in accepted}),
                  note="Small corpora may have empty splits; source independence takes priority over ratios.")
    write_json(output / "report.json", report)
    return report
