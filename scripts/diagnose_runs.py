"""python scripts/diagnose_runs.py RUN [RUN ...] --output EMPTY_DIR"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from codes.multihop_qa.diagnostics import diagnose_runs

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(diagnose_runs(args.folders,args.output),ensure_ascii=False,indent=2))
