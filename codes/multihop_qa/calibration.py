"""Select necessity thresholds exclusively on source-disjoint human validation labels."""
from .schema import empty_output, write_json, probability


def classification(rows, threshold):
    tp=fp=tn=fn=0
    for row in rows:
        predicted = row['score'] > threshold
        label = row['label'] == 1
        tp += predicted and label
        fp += predicted and not label
        tn += not predicted and not label
        fn += not predicted and label
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,precision=precision,recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def calibrate(rows, output, min_precision=.9):
    probability(min_precision,'minimum precision')
    groups, ids = {}, set()
    for row in rows:
        probability(row.get('score'),'necessity score')
        if row.get('human_reviewed') is not True or not row.get('reviewer'):
            raise ValueError('human labels required for calibration')
        if type(row.get('label')) is not int or row['label'] not in {0,1} or row.get('split') not in {'val','test'}:
            raise ValueError('calibration requires labelled val/test records')
        if row['record_id'] in ids:
            raise ValueError('duplicate calibration sample')
        ids.add(row['record_id'])
        if not row.get('source_group_ids'):
            raise ValueError('source groups required')
        for group in row['source_group_ids']:
            if group in groups and groups[group] != row['split']:
                raise ValueError('validation/test source leakage')
            groups[group] = row['split']
    validation = [r for r in rows if r['split']=='val']
    if {r['label'] for r in validation} != {0,1}:
        raise ValueError('both positive and negative human validation labels required')
    candidates = sorted({0.,1.} | {r['score'] for r in validation})
    viable = [(t,classification(validation,t)) for t in candidates]
    viable = [(t,m) for t,m in viable if m['precision'] >= min_precision and m['tp']+m['fp']>0]
    if not viable:
        raise ValueError('no threshold meets requested validation precision')
    threshold,metric = max(viable,key=lambda item:(item[1]['f1'],item[1]['precision'],item[0]))
    test = [r for r in rows if r['split']=='test']
    report = dict(threshold=threshold,comparison='score > threshold',calibrated=True,
                  validation_samples=len(validation),validation=metric,
                  test_samples=len(test),test=classification(test,threshold) if test else None,
                  note='Threshold selection uses validation only; no test labels used for tuning.')
    output = empty_output(output)
    write_json(output/'calibration.json',report)
    return report
