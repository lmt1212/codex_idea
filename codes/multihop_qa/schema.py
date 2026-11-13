"""Versioned records, stable identifiers and fail-closed validation helpers."""
import hashlib
import json
import math
from pathlib import Path

VERSION = "0.1.0"
ROLES = {"adoption", "capability", "problem", "method", "condition", "limitation",
         "cause", "mechanism", "effect", "attribute", "baseline", "improvement", "conclusion"}


def uid(kind, value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return kind + ":" + hashlib.sha256(encoded).hexdigest()[:24]


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty string required: " + label)
    return value


def strings(value, label, nonempty=False):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError("string list required: " + label)
    if nonempty and not value:
        raise ValueError("nonempty list required: " + label)
    if len(value) != len(set(value)):
        raise ValueError("duplicate values: " + label)
    return value


def probability(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score outside [0,1]: " + label)
    return float(value)


def read_jsonl(path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: object required")
            records.append(value)
    return records


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def empty_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("use a new or empty output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def verification_fingerprint(qa, chunks):
    fields = ("qa_id", "path_id", "question", "answer", "answer_claims", "fact_ids", "evidence_chunk_ids", "source_group_ids")
    return uid("verified", [dict((k, qa[k]) for k in fields),
                             [chunks[c] for c in qa["evidence_chunk_ids"]]])


def require_external_provider_allowed():
    policy=Path(__file__).resolve().parents[2]/'project_policy.json'
    if policy.exists() and json.loads(policy.read_text(encoding='utf8')).get('local_models_only') is True:
        raise ValueError('project policy permits local models only; external document API calls are disabled')
