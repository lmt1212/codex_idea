"""Annotation-based metrics; no self-generated gold labels."""


def precision_recall(predicted, gold):
    p, g = set(predicted), set(gold)
    correct = len(p & g)
    return dict(precision=correct / len(p) if p else 0., recall=correct / len(g) if g else 0.,
                true_positive=correct, predicted=len(p), gold=len(g))


def human_quality(reviews):
    eligible = [r for r in reviews if r.get("human_reviewed") is True]
    if not eligible:
        return dict(reviewed=0, acceptance_rate=None, true_multihop_ratio=None)
    return dict(reviewed=len(eligible),
                acceptance_rate=sum(r.get("decision") == "approved" for r in eligible) / len(eligible),
                true_multihop_ratio=sum(r.get("true_multihop") is True for r in eligible) / len(eligible))
