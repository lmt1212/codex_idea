"""Bounded extraction diagnosis on real text; never writes human labels."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from codes.multihop_qa.providers import LocalJSONClient
from codes.multihop_qa.schema import read_jsonl,empty_output,write_json

p=argparse.ArgumentParser()
p.add_argument('input',type=Path)
p.add_argument('--model',required=True)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
rows=[c for c in read_jsonl(a.input/'chunks.jsonl') if c.get('eligible_for_experiment') and not c['needs_review']]
c=LocalJSONClient(a.model,max_calls=6)
results=[]
for row in rows[:2]:
    for prompt in [
        '请从原文提取一个明确事实。只返回JSON：{"subject":"主体","predicate":"关系","object":"客体","quote":"逐字原文"}。',
        '请从原文选择一个完整句子，原样复制。只返回JSON：{"quote":"逐字原句"}。']:
        try:
            result=c.ask('extraction_diagnosis',prompt,dict(text=row['text']))
            results.append(dict(chunk_id=row['chunk_id'],result=result,quote_matched=isinstance(result.get('quote'),str) and bool(result['quote']) and result['quote'] in row['text']))
        except Exception as ex:
            results.append(dict(chunk_id=row['chunk_id'],error=str(ex)))
out=empty_output(a.output)
write_json(out/'report.json',dict(results=results,usage=c.history))
print(json.dumps(results,ensure_ascii=False))
