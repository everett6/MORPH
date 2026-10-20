"""Collect kernel-checked declaration-reference traces from pinned Mathlib.

These are theorem re-checks by reference, not LeanDojo multi-step trajectories
or discoveries. The target declaration name is never included in the prompt.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from morph.graph import ProofGraph,ProofNode
from morph.inference_payload import payload_prompt
from verifier import LeanClient


def collect(project,output,limit=1000,workers=4,max_bytes=120,report=None):
    client=LeanClient(project=project,imports=('Mathlib.Data.Nat.Basic','Mathlib.Logic.Basic'),timeout=30)
    export=Path(__file__).resolve().parents[1]/'verifier/Export.lean'
    client.timeout=180
    code,text=client._run(['-j1',str(export)])
    if code:raise RuntimeError(text[-4000:])
    client.timeout=30
    candidates=[];seen=set()
    for line in text.splitlines():
        if not line.startswith('{'):continue
        record=json.loads(line);statement=' '.join(record['statement'].split());name=record['name']
        if name.split('.')[0] not in {'Nat','Int','Bool','BitVec','Fin','List','Option','Prod','Set'}:continue
        if len(statement.encode())>max_bytes or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_'.]*",name):continue
        family=hashlib.sha256(statement.encode()).hexdigest()
        if family in seen:continue
        seen.add(family);candidates.append((name,statement,family))
    candidates.sort(key=lambda row:row[2])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    accepted=[];rejected=[]
    def check(item):
        name,statement,family=item;proof=f'by exact @{name}'
        receipt=client.verify(statement,proof)
        return item,proof,receipt
    # Bounded batches avoid launching unneeded thousands of compiler processes.
    with output.open('w') as handle,ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0,len(candidates),workers*8):
            for (name,statement,family),proof,receipt in pool.map(check,candidates[start:start+workers*8]):
                if len(accepted)>=limit:break
                if not receipt.success:
                    rejected.append(dict(name=name,diagnostics=receipt.diagnostics));continue
                bucket=int(family[:8],16)%10;split='validation' if bucket==0 else 'test' if bucket==1 else 'train'
                graph=ProofGraph();graph.add(ProofNode(0,statement,formal_statement=statement))
                row=dict(family=family,split=split,text=payload_prompt(statement)+json.dumps(dict(proof=proof),ensure_ascii=True)+'\0',
                         graph=graph.snapshot(),action=dict(type='VERIFY',target=0),value=1.,candidate_proof=proof,
                         verification=asdict(receipt),trace_kind='declaration_reference',
                         source=dict(declaration=name,mathlib_revision=client.project_revision,imports=list(client.imports),normalized_type_sha256=family))
                handle.write(json.dumps(row,ensure_ascii=False)+'\n');accepted.append(row)
            handle.flush()
            if len(accepted)>=limit:break
    summary=dict(accepted=len(accepted),rejected=len(rejected),trace_kind='declaration_reference',
                 split_counts={split:sum(row['split']==split for row in accepted) for split in ['train','validation','test']},
                 mathlib_revision=client.project_revision,lean_version=accepted[0]['verification']['version'] if accepted else None,
                 isolated=True,dataset_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                 alias_policy='exact whitespace-normalized type hash; not semantic equivalence',
                 verified_new_theorems=False,lean_dojo_multistep=False,rejection_examples=rejected[:5])
    if report:Path(report).write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',default='.tools/mathlib4')
    parser.add_argument('--output',default='.generated/verified_declarations.jsonl');parser.add_argument('--limit',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--report',default='docs/verified_data_report.json')
    args=parser.parse_args();print(json.dumps(collect(args.project,args.output,args.limit,args.workers,report=args.report)))


if __name__=='__main__':main()
