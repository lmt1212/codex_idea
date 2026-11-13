"""Select a bounded, explicitly unreviewed real-book pilot without changing source text."""
import argparse
import sys
import re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from codes.multihop_qa.schema import empty_output,read_jsonl,write_jsonl,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-document',type=int,default=12)
    args=parser.parse_args()
    chunks=read_jsonl(args.input/'chunks.jsonl')
    counts={}
    for c in chunks:
        count=counts.get(c['doc_id'],0)
        content=c['text'].strip()
        chinese=len(re.findall(r'[\u4e00-\u9fff]',content))
        metadata=bool(re.search(r'University|Department|ISBN|版权|收稿日期|作者简介|基金项目|出版发行|印刷|责任编辑',content,re.I))
        usable=(not c['needs_review'] and c['kind']=='text' and 60<=len(content)<=900
                and chinese/len(content)>.4 and content.endswith('。') and not metadata
                and count<args.per_document)
        c['eligible_for_experiment']=usable
        if usable:
            counts[c['doc_id']]=count+1
    output=empty_output(args.output)
    write_jsonl(output/'documents.jsonl',read_jsonl(args.input/'documents.jsonl'))
    write_jsonl(output/'chunks.jsonl',chunks)
    report=dict(selected=sum(counts.values()),per_document=counts,human_reviewed=False,
                note='Length/layout eligibility only. Requires original-page quality review before acceptance.')
    write_json(output/'report.json',report)
    print(report)


if __name__=='__main__':
    main()
