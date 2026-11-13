"""Persisted stages sharing a single schema; each run writes to a fresh folder."""
from pathlib import Path
import json
from collections import Counter
from .schema import VERSION, read_jsonl, write_jsonl, write_json, empty_output
from .ingestion import SUPPORTED, parse_file, chunk_documents
from .extraction import extract, validate_fact, validate_cached_fact
from .retrieval import CandidateIndex
from .mining import mine_paths
from .graph import build_graph
from .generation import generate_qa


def load(folder, name):
    return read_jsonl(Path(folder) / (name + ".jsonl"))


def persist(output, records, report):
    for name, rows in records.items():
        write_jsonl(output / (name + ".jsonl"), rows)
    write_json(output / "report.json", dict(pipeline_version=VERSION, **report))


def ingest_stage(source, output, max_chars=1200, source_groups=None, pdf_backend='text', pages=None, ocr='auto', provisional=False):
    source = Path(source)
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED) if source.is_dir() else [source]
    if not files:
        raise ValueError("no supported source files")
    output = empty_output(output)
    docs, errors, seen = [], [], set()
    for path in files:
        try:
            group = (source_groups or {}).get(str(path.resolve()))
            doc = parse_file(path, group, pdf_backend, pages, ocr, provisional)
            if doc["doc_id"] not in seen:
                docs.append(doc)
                seen.add(doc["doc_id"])
        except Exception as ex:
            errors.append(dict(source=str(path), error_type=type(ex).__name__, reason=str(ex)))
    chunks = chunk_documents(docs, max_chars)
    report = dict(stage="ingest", input_files=len(files), documents=len(docs), chunks=len(chunks),
                  review_chunks=sum(c["needs_review"] for c in chunks), errors=len(errors),
                  parser_backend=pdf_backend,ocr_mode=ocr,provisional_native=provisional,
                  limitation="Automated OCR/layout results require independent source review; no claim of full-book accuracy.")
    persist(output, dict(documents=docs, chunks=chunks, errors=errors), report)
    return report


def extraction_stage(folder, output, client=None, cached_facts=None, source_reviews=None, aliases=None, limit=None):
    chunks, docs = load(folder, "chunks"), load(folder, "documents")
    reviews = {r["chunk_id"]: r for r in (source_reviews or [])}
    for chunk in chunks:
        review = reviews.get(chunk["chunk_id"], {})
        if review.get("decision") == "approved" and review.get("human_reviewed") is True and review.get("reviewer"):
            chunk["needs_review"] = False
            chunk["source_review"] = review
    by_chunk = {c["chunk_id"]: c for c in chunks}
    facts, rejected, errors, seen = [], [], [], set()
    if cached_facts is None and client is None:
        raise ValueError("provide a fact cache or an explicitly enabled extraction client")
    output = empty_output(output)
    if cached_facts is not None:
        for raw in cached_facts:
            try:
                chunk = by_chunk[raw["chunk_id"]]
                if chunk["needs_review"]:
                    raise ValueError("source requires review")
                f = validate_fact(raw, chunk, aliases)
                if f["fact_id"] not in seen:
                    facts.append(f)
                    seen.add(f["fact_id"])
            except (ValueError, KeyError, TypeError) as ex:
                rejected.append(dict(chunk_id=raw.get("chunk_id"), reason=str(ex)))
    else:
        selected = [c for c in chunks if not c["needs_review"] and c.get('eligible_for_experiment',True)]
        if limit is not None:
            selected = selected[:limit]
        for chunk in selected:
            try:
                for f in extract(chunk, client, aliases):
                    if f["fact_id"] not in seen:
                        facts.append(f)
                        seen.add(f["fact_id"])
            except Exception as ex:
                errors.append(dict(chunk_id=chunk["chunk_id"], error_type=type(ex).__name__,
                                   reason=str(ex) if isinstance(ex, ValueError) else "extraction_request_failed"))
            write_jsonl(output / "facts.jsonl", facts)
            write_jsonl(output / "calls.jsonl", client.history)
    report = dict(stage="extraction", chunks=len(chunks), facts=len(facts), rejected=len(rejected),
                  errors=len(errors), review_chunks=sum(c["needs_review"] for c in chunks),
                  extraction_mode="cache" if cached_facts is not None else ('local' if type(client).__name__=='LocalJSONClient' else 'api'),
                  usage=client.history if client else [], status="facts_pending_semantic_verification")
    if cached_facts is None:
        report.update(eligible_chunks=sum(not c['needs_review'] and c.get('eligible_for_experiment',True) for c in chunks),
                      attempted_chunks=len(selected),
                      chunks_with_facts=len({f['chunk_id'] for f in facts}),
                      empty_chunks=len(selected)-len(errors)-len({f['chunk_id'] for f in facts}))
    write_json(output / "aliases.json", aliases or {})
    persist(output, dict(documents=docs, chunks=chunks, facts=facts, rejected=rejected, errors=errors), report)
    return report


def mining_stage(folder, output, embeddings=None, beam_width=16, path_limit=100, per_channel=32, strategy="beam", scorer=None, implicit=False):
    docs, chunks, facts = (load(folder, name) for name in ("documents", "chunks", "facts"))
    # Re-check cache ownership; never trust stale IDs and coordinates from disk.
    chunk_map = {c["chunk_id"]: c for c in chunks}
    alias_file = Path(folder) / "aliases.json"
    aliases = json.loads(alias_file.read_text(encoding="utf8")) if alias_file.exists() else {}
    for f in facts:
        validate_cached_fact(f, chunk_map[f["chunk_id"]], aliases)
    index = CandidateIndex(facts, embeddings, per_channel, scorer, implicit)
    candidates = [edge for fact in facts for edge in index.recall(fact)]
    paths = mine_paths(facts, candidates, beam_width, path_limit, strategy)
    graph = build_graph(docs, chunks, facts, candidates)
    output = empty_output(output)
    report = dict(stage="mining", facts=len(facts), candidate_edges=len(candidates), paths=len(paths),
                  semantic_recall=bool(embeddings), implicit_role_recall=implicit, search=strategy, errors=0,
                  scoring="trained_linear" if scorer else "rule_based", candidate_steps_are_verified_hops=False)
    write_json(output / "aliases.json", aliases)
    persist(output, dict(documents=docs, chunks=chunks, facts=facts, candidates=candidates, paths=paths,
                         graph_nodes=graph["nodes"], graph_edges=graph["edges"]), report)
    return report


def generation_stage(folder, output, generator, verifier, threshold=.1, limit=10, calibration=None, review_mode='independent'):
    if generator.model == verifier.model and review_mode!='same-model-diagnostic':
        raise ValueError("generation and verification clients must differ")
    if limit < 1:
        raise ValueError("positive generation limit required")
    docs, chunks, facts, paths = (load(folder, name) for name in ("documents", "chunks", "facts", "paths"))
    fact_map = {f["fact_id"]: f for f in facts}
    chunk_map = {c["chunk_id"]: c for c in chunks}
    alias_file = Path(folder) / "aliases.json"
    aliases = json.loads(alias_file.read_text(encoding="utf8")) if alias_file.exists() else {}
    for fact in facts:
        validate_cached_fact(fact, chunk_map[fact["chunk_id"]], aliases)
    output = empty_output(output)
    qas, rejected, errors, signatures = [], [], [], set()
    for path in paths[:limit]:
        try:
            qa, rejection = generate_qa(path, fact_map, chunk_map, generator, verifier, threshold,review_mode)
            if qa:
                signature = ("".join(qa["question"].split()), "".join(qa["answer"].split()))
                if signature in signatures:
                    rejected.append(dict(stage="deduplication", path_id=path["path_id"], reason="duplicate_qa"))
                else:
                    signatures.add(signature)
                    qas.append(qa)
            if rejection:
                rejected.append(rejection)
        except Exception as ex:
            errors.append(dict(path_id=path["path_id"], error_type=type(ex).__name__,
                               reason=str(ex) if isinstance(ex, ValueError) else "generation_or_verification_failed"))
        for name, rows in (("qa", qas), ("rejected", rejected), ("errors", errors)):
            write_jsonl(output / (name + ".jsonl"), rows)
        write_jsonl(output / 'generator_calls.jsonl', generator.history)
        write_jsonl(output / 'verifier_calls.jsonl', verifier.history)
    report = dict(stage="generation", candidate_paths=len(paths), attempted=min(limit, len(paths)),
                  auto_passed=sum(q['verification_status']=='auto_passed_pending_human_review' for q in qas),
                  diagnostic_candidates=sum(q['verification_status']=='diagnostic_pending_independent_and_human_review' for q in qas),
                  review_mode=review_mode,rejected=len(rejected), errors=len(errors), human_accepted=0,
                  necessity_threshold=threshold, threshold_calibrated=bool(calibration),calibration=calibration,
                  generator_usage=generator.history, verifier_usage=verifier.history,
                  limitation="Model verification is not human proof; no claimed dataset acceptance rate.")
    report['rejections_by_stage']=dict(Counter(r.get('stage','unknown') for r in rejected))
    report['verification_calls_by_task']=dict(Counter(r.get('task','unknown') for r in verifier.history))
    persist(output, dict(documents=docs, chunks=chunks, facts=facts, paths=paths, qa=qas, rejected=rejected, errors=errors), report)
    return report
