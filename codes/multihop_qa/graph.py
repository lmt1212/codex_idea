"""Persist a multi-granularity graph without upgrading candidate links to proof."""
import re
from .schema import uid


def build_graph(documents, chunks, facts, candidates):
    nodes, edges, seen = [], [], set()
    def node(identifier, kind, **metadata):
        if identifier not in seen:
            nodes.append(dict(id=identifier, type=kind, **metadata))
            seen.add(identifier)
    for doc in documents:
        node(doc["doc_id"], "Document", title=doc["title"], source=doc["source"], source_group_id=doc["source_group_id"])
    previous = {}
    sentences = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        node(cid, "Chunk", source_position=chunk["source_position"], spans=chunk["spans"])
        edges.append(dict(source=chunk["doc_id"], target=cid, type="CONTAINS"))
        if chunk["doc_id"] in previous:
            edges.append(dict(source=previous[chunk["doc_id"]], target=cid, type="NEXT"))
        previous[chunk["doc_id"]] = cid
        sentences[cid] = []
        for m in re.finditer(r"[^。！？\n]+[。！？]?", chunk["text"]):
            sid = uid("sentence", [cid, m.start(), m.end()])
            node(sid, "Sentence", chunk_id=cid, start=m.start(), end=m.end(), quote=m[0])
            sentences[cid].append((sid, m.start(), m.end()))
            edges.append(dict(source=cid, target=sid, type="CONTAINS"))
    for fact in facts:
        fid, cid = fact["fact_id"], fact["chunk_id"]
        node(fid, "Fact", role=fact["role"], status=fact["status"])
        for field in ("subject_id", "object_id"):
            eid = uid("entity", fact[field])
            node(eid, "Entity", canonical_name=fact[field])
            edges.append(dict(source=cid, target=eid, type="ENTITY_LINK", fact_id=fid))
        edges.append(dict(source=uid("entity", fact["subject_id"]), target=uid("entity", fact["object_id"]),
                          type="RELATION_LINK", fact_id=fid, predicate=fact["predicate"],
                          conditions=fact["conditions"], time=fact["time"], polarity=fact["polarity"],
                          modality=fact["modality"], status="candidate"))
        edges.append(dict(source=cid, target=fid, type="SUPPORT", status="quote_matched_only"))
        for sid, start, end in sentences[cid]:
            if start <= fact["start"] and fact["end"] <= end:
                edges.append(dict(source=sid, target=fid, type="SUPPORT", status="quote_matched_only"))
        if fact["event"]:
            eid = uid("event", [fact["event"], fact["time"], fact["subject_id"]])
            node(eid, "Event", name=fact["event"], time=fact["time"])
            edges.append(dict(source=fid, target=eid, type="EVENT_LINK"))
    for candidate in candidates:
        for kind, enabled in (("SEMANTIC_LINK", "semantic" in candidate["channels"]),
                              ("RELATION_COMPLEMENT", candidate["scores"]["complement"] > 0),
                              ("CONTRAST", "comparison" in candidate["channels"])):
            if enabled:
                edges.append(dict(source=candidate["source"], target=candidate["target"], type=kind,
                                  scores=candidate["scores"], status="candidate"))
    return dict(nodes=nodes, edges=edges)
