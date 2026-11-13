"""Optional supervised linear complement scorer using independently labelled pairs."""
import math
from .schema import empty_output, write_json, probability

FEATURES = ("entity", "relation", "semantic", "complement")


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def predict(features, model):
    return sigmoid(model["bias"] + sum(model["weights"][k] * features[k] for k in FEATURES))


def metrics(rows, model):
    loss, correct = 0., 0
    for row in rows:
        score = predict(row["features"], model)
        score = min(1-1e-12, max(1e-12, score))
        label = row["label"]
        loss += -label*math.log(score) - (1-label)*math.log(1-score)
        correct += (score >= .5) == bool(label)
    return dict(samples=len(rows), log_loss=loss/len(rows) if rows else None,
                accuracy=correct/len(rows) if rows else None)


def train_scorer(annotations, output, epochs=500, learning_rate=.1, regularization=.01):
    if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(regularization) or regularization < 0:
        raise ValueError("invalid training parameters")
    groups, seen = {}, set()
    splits = {s: [] for s in ("train", "val", "test")}
    for row in annotations:
        if row.get("human_reviewed") is not True or not row.get("reviewer"):
            raise ValueError("training requires independently human-labelled annotations")
        if row.get("split") not in splits or type(row.get("label")) is not int or row["label"] not in {0, 1}:
            raise ValueError("invalid split or binary label")
        if not row.get("source_group_ids") or not isinstance(row["source_group_ids"], list):
            raise ValueError("source groups required for training")
        pair = tuple(sorted(row["fact_ids"]))
        if len(pair) != 2 or pair in seen:
            raise ValueError("duplicate or invalid labelled fact pair")
        seen.add(pair)
        for group in row["source_group_ids"]:
            if group in groups and groups[group] != row["split"]:
                raise ValueError("training source-group leakage")
            groups[group] = row["split"]
        for name in FEATURES:
            probability(row["features"].get(name), name)
        splits[row["split"]].append(row)
    if {r["label"] for r in splits["train"]} != {0, 1} or not splits["val"]:
        raise ValueError("need both training classes and a source-disjoint validation set")
    model = dict(model_type="linear_complementarity", feature_names=list(FEATURES),
                 weights={name: 0. for name in FEATURES}, bias=0.)
    for _ in range(epochs):
        gradient = dict.fromkeys(FEATURES, 0.)
        bias_gradient = 0.
        for row in splits["train"]:
            error = predict(row["features"], model) - row["label"]
            for name in FEATURES:
                gradient[name] += error*row["features"][name]
            bias_gradient += error
        count = len(splits["train"])
        for name in FEATURES:
            model["weights"][name] -= learning_rate*(gradient[name]/count + regularization*model["weights"][name])
        model["bias"] -= learning_rate*bias_gradient/count
    report = dict(epochs=epochs, metrics={s: metrics(rows, model) for s, rows in splits.items()},
                  note="Supervised linear baseline, not a validated neural/contrastive research contribution.")
    output = empty_output(output)
    write_json(output / "scorer.json", model)
    write_json(output / "report.json", report)
    return report
