"""Relation-role-constrained beam search and explainable global path ranking."""
from collections import defaultdict
from .retrieval import TEMPLATES, entities, compatible
from .schema import uid


def score_path(path, roles, links):
    used_roles = [f["role"] for f in path]
    coverage = 1. if roles == ("*", "*") else len(path) / len(roles)
    coherence = sum(links) / max(1, len(links)) if links else 0.
    diversity = len({f["source_group_id"] for f in path}) / len(path)
    information = len(set(used_roles)) / len(roles) if "*" not in roles else len({f["predicate"] for f in path}) / len(path)
    signatures = {(f["subject_id"], f["predicate"], f["object_id"]) for f in path}
    redundancy = 1 - len(signatures) / len(path)
    components = dict(coherence=coherence, coverage=coverage, diversity=diversity,
                      complementarity=information, redundancy=redundancy)
    score = .3*coherence + .25*coverage + .15*diversity + .3*information - .25*redundancy
    return score, components


def mine_paths(facts, candidate_edges, beam_width=16, path_limit=100, strategy="beam"):
    if beam_width < 1 or path_limit < 1 or strategy not in {"beam", "bfs"}:
        raise ValueError("invalid search configuration")
    by_id = {f["fact_id"]: f for f in facts}
    adjacency = defaultdict(dict)
    for edge in candidate_edges:
        adjacency[edge["source"]][edge["target"]] = edge
        reverse = dict(edge, source=edge["target"], target=edge["source"])
        adjacency[edge["target"]].setdefault(edge["source"], reverse)
    completed = {}
    for task, roles in TEMPLATES.items():
        for seed in facts:
            if roles[0] != "*" and seed["role"] != roles[0]:
                continue
            if seed["polarity"] != "affirmed" or seed["modality"] != "certain":
                continue
            beam = [([seed], [])]
            for depth in range(1, len(roles)):
                expanded = []
                for path, links in beam:
                    used_chunks = {f["chunk_id"] for f in path}
                    neighbor_ids = {fid for f in path for fid in adjacency[f["fact_id"]]}
                    for fid in sorted(neighbor_ids):
                        f = by_id[fid]
                        if f["chunk_id"] in used_chunks or (roles[depth] != "*" and f["role"] != roles[depth]):
                            continue
                        if not all(compatible(f, other) for other in path):
                            continue
                        if task.startswith("entity_bridge"):
                            if path[-1]["object_id"] != f["subject_id"]:
                                continue
                            visited = {path[0]["subject_id"]} | {other["object_id"] for other in path}
                            if f["object_id"] in visited:
                                continue
                        if task == "comparison" and (path[0]["predicate"] != f["predicate"] or path[0]["subject_id"] == f["subject_id"] or path[0]["conditions"] != f["conditions"]):
                            continue
                        if task == 'comparison' and path[0]['object_id']==f['object_id']:
                            # Identical extracted values do not establish a contrast;
                            # keep their edges for other task types, not this template.
                            continue
                        connections = [adjacency[other["fact_id"]][fid] for other in path if fid in adjacency[other["fact_id"]]]
                        # Semantic recall alone is not a reasoning relation. Require
                        # a bridge or a comparable shared attribute before expansion.
                        valid = [e for e in connections if e["scores"]["entity"] > 0
                                 or (task == "comparison" and "comparison" in e["channels"])
                                 or (not task.startswith('entity_bridge') and e['scores']['complement'] > 0
                                     and ('semantic' in e['channels'] or 'implicit_role' in e['channels']))]
                        if not valid:
                            continue
                        edge_score = max(e["score"] for e in valid)
                        expanded.append((path + [f], links + [edge_score]))
                expanded.sort(key=lambda item: (-score_path(item[0], roles, item[1])[0], tuple(f["fact_id"] for f in item[0])))
                # BFS is bounded explicitly; suitable only as a small-data control.
                beam = expanded[:beam_width] if strategy == "beam" else expanded[:beam_width*16]
            for path, links in beam:
                if len(path) != len(roles):
                    continue
                score, components = score_path(path, roles, links)
                signature = [task, sorted(f["fact_id"] for f in path)]
                pid = uid("path", signature)
                completed[pid] = dict(path_id=pid, task_type=task, fact_ids=[f["fact_id"] for f in path],
                                     evidence_chunk_ids=[f["chunk_id"] for f in path],
                                     source_group_ids=sorted({f["source_group_id"] for f in path}),
                                     cross_document=len({f["doc_id"] for f in path}) > 1,
                                     candidate_steps=len(path), hop_count=None,
                                     score=score, score_components=components,
                                     status="candidate_pending_semantic_verification")
    # Rank all retained candidates before the output budget, not by input order.
    return sorted(completed.values(), key=lambda p: (-p["score"], p["path_id"]))[:path_limit]
