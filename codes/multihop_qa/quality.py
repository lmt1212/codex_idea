"""Real quality reports use blinded human responses, not fixture/model self-labels."""
from collections import defaultdict
from .annotations import completed_reviews, QA_FIELDS
from .schema import empty_output, write_json, write_jsonl


def quality_report(pack, output):
    rows, total = completed_reviews(pack)
    methods = defaultdict(list)
    for row in rows:
        if row['kind'] == 'qa':
            methods[row['method']].append(row)
    report = dict(items=total,reviewed=len(rows),review_coverage=len(rows)/total if total else 0.,
                  status='measured_on_human_labels' if rows else 'awaiting_human_annotation',methods={})
    for name,items in methods.items():
        report['methods'][name] = dict(reviewed=len(items),
            **{field:sum(r[field] for r in items)/len(items) for field in QA_FIELDS},
            acceptance_rate=sum(r['decision']=='approved' for r in items)/len(items))
    output = empty_output(output)
    write_json(output/'report.json',report)
    write_jsonl(output/'reviews.jsonl',[dict(qa_id=r['qa_id'],decision=r['decision'],human_reviewed=True,
                 reviewer=r['reviewer'],hop_count=r['hop_count'],**{k:r[k] for k in QA_FIELDS})
                 for r in rows if r['kind']=='qa'])
    write_jsonl(output/'pair_labels.jsonl',[dict(fact_ids=r['fact_ids'],features=r['features'],label=r['label'],
                 source_group_ids=r['source_group_ids'],split=r['split'],reviewer=r['reviewer'],human_reviewed=True)
                 for r in rows if r['kind']=='pair'])
    write_jsonl(output/'calibration_samples.jsonl',[dict(record_id=r['record_id'],score=r['necessity_score'],
                 label=int(r['true_multihop']),split=r['split'],source_group_ids=r['source_group_ids'],
                 reviewer=r['reviewer'],human_reviewed=True)
                 for r in rows if r['kind']=='qa' and isinstance(r.get('necessity_score'),(float,int))])
    return report
