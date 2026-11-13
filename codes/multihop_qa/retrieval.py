"""Indexed entity/relation/role recall and optional bounded vector-LSH recall."""
import math
import random
from collections import defaultdict
from .schema import probability

TEMPLATES = {
    "problem_solution": ("adoption", "capability", "problem"),
    "method_evaluation": ("method", "condition", "limitation"),
    "causal_explanation": ("cause", "mechanism", "effect"),
    "comparison": ("attribute", "attribute"),
    "conditional_reasoning": ("condition", "conclusion"),
    "technical_evolution": ("baseline", "limitation", "improvement"),
    "entity_bridge": ("*", "*"),
    "entity_bridge_3": ("*", "*", "*"),
    "entity_bridge_4": ("*", "*", "*", "*"),
}
ROLE_PAIRS = {frozenset((a, b)) for roles in TEMPLATES.values() if "*" not in roles
              for a in roles for b in roles if a != b}


def normalize_vector(vector):
    if not isinstance(vector, list) or not vector or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in vector):
        raise ValueError("invalid embedding")
    norm = math.sqrt(sum(x*x for x in vector))
    if not norm:
        raise ValueError("zero embedding")
    return [x / norm for x in vector]


class VectorLSH:
    """Approximate candidates only; rerank by cosine. No full pairwise scan."""
    def __init__(self, vectors, tables=4, bits=8, bucket_limit=128):
        self.vectors = {k: normalize_vector(v) for k, v in vectors.items()}
        self.dimension = len(next(iter(self.vectors.values()))) if self.vectors else 0
        if any(len(v) != self.dimension for v in self.vectors.values()):
            raise ValueError("embedding dimensions differ")
        self.bucket_limit = bucket_limit
        rng = random.Random(2026)
        self.planes = [[[rng.gauss(0, 1) for _ in range(self.dimension)] for _ in range(bits)] for _ in range(tables)]
        self.buckets = [defaultdict(list) for _ in self.planes]
        for key in sorted(self.vectors):
            for planes, buckets in zip(self.planes, self.buckets):
                buckets[self.signature(self.vectors[key], planes)].append(key)

    @staticmethod
    def signature(vector, planes):
        return tuple(sum(x*y for x, y in zip(vector, plane)) >= 0 for plane in planes)

    def query(self, key, k):
        if key not in self.vectors:
            return []
        vector, candidates = self.vectors[key], set()
        for planes, buckets in zip(self.planes, self.buckets):
            candidates.update(buckets.get(self.signature(vector, planes), [])[:self.bucket_limit])
        scored = [(other, max(0., sum(a*b for a, b in zip(vector, self.vectors[other]))))
                  for other in candidates if other != key]
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:k]


def entities(fact):
    return {fact["subject_id"], fact["object_id"]}


def compatible(a, b):
    if a["chunk_id"] == b["chunk_id"]:
        return False
    if (a["subject_id"], a["predicate"], a["object_id"]) == (b["subject_id"], b["predicate"], b["object_id"]):
        return False
    if a["polarity"] != "affirmed" or b["polarity"] != "affirmed" or a["modality"] != "certain" or b["modality"] != "certain":
        return False
    if a["time"] and b["time"] and a["time"] != b["time"]:
        return False
    # Different explicit values for the same guard key conflict; other guards are
    # retained for semantic review, never assumed equivalent.
    def guards(f):
        return dict(c.split("=", 1) for c in f["conditions"] if "=" in c)
    left, right = guards(a), guards(b)
    return not any(left[k] != right[k] for k in left.keys() & right.keys())


class CandidateIndex:
    def __init__(self, facts, embeddings=None, per_channel=32, scorer=None, implicit=False):
        if per_channel < 1:
            raise ValueError("positive recall budget required")
        self.facts = {f["fact_id"]: f for f in facts}
        self.entity = defaultdict(list)
        self.subject = defaultdict(list)
        self.role = defaultdict(list)
        self.global_role = defaultdict(list)
        self.predicate = defaultdict(list)
        self.chunk = defaultdict(list)
        self.limit = per_channel
        self.scorer = scorer
        self.implicit = implicit
        if scorer is not None:
            if scorer.get("model_type") != "linear_complementarity":
                raise ValueError("unknown scorer model type")
            values = [scorer.get("bias")] + [scorer.get("weights", {}).get(k) for k in ("entity", "relation", "semantic", "complement")]
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
                raise ValueError("invalid scorer weights")
        for f in sorted(facts, key=lambda f: f["fact_id"]):
            fid = f["fact_id"]
            self.global_role[f['role']].append(fid)
            for e in entities(f):
                self.entity[e].append(fid)
                self.role[(e, f["role"])].append(fid)
            self.subject[f["subject_id"]].append(fid)
            self.predicate[f["predicate"]].append(fid)
            self.chunk[f["chunk_id"]].append(fid)
        self.vectors = VectorLSH(embeddings or {})

    def recall(self, fact):
        candidates = defaultdict(set)
        semantic = {}
        def add(ids, channel):
            for fid in ids[:self.limit]:
                if fid != fact["fact_id"]:
                    candidates[fid].add(channel)
        for e in sorted(entities(fact)):
            add(self.entity[e], "entity")
            for role in sorted({r for pair in ROLE_PAIRS if fact["role"] in pair for r in pair if r != fact["role"]}):
                add(self.role[(e, role)], "complement")
        if self.implicit:
            complementary_roles = {r for pair in ROLE_PAIRS if fact['role'] in pair for r in pair if r != fact['role']}
            for role in sorted(complementary_roles):
                add(self.global_role[role], 'implicit_role')
        add(self.subject[fact["object_id"]], "relation")
        if fact["role"] == "attribute":
            add(self.predicate[fact["predicate"]], "comparison")
        for cid, similarity in self.vectors.query(fact["chunk_id"], self.limit):
            for fid in self.chunk[cid][:self.limit]:
                candidates[fid].add("semantic")
                semantic[fid] = similarity
        results = []
        for fid, channels in candidates.items():
            target = self.facts[fid]
            if not compatible(fact, target):
                continue
            overlap = len(entities(fact) & entities(target)) / len(entities(fact) | entities(target))
            relation = float(fact["object_id"] == target["subject_id"])
            comparison = fact["role"] == target["role"] == "attribute" and fact["predicate"] == target["predicate"] and fact["subject_id"] != target["subject_id"]
            complement = float(frozenset((fact["role"], target["role"])) in ROLE_PAIRS or comparison)
            scores = dict(entity=overlap, relation=relation, semantic=semantic.get(fid, 0.), complement=complement)
            score = .2*overlap + .25*relation + .15*scores["semantic"] + .4*complement
            if self.scorer is not None:
                from .training import predict
                score = predict(scores, self.scorer)
            results.append(dict(source=fact["fact_id"], target=fid, channels=sorted(channels),
                                scores=scores, score=score, status="candidate",
                                requires_semantic_bridge_review=overlap == 0 and not comparison))
        return sorted(results, key=lambda r: (-r["score"], r["target"]))
