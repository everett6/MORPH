"""Train a tiny complete model on real-checked toy proofs and log its bytes.

No payload provider or runtime tactic fallback. Repair-template data is labeled
as such; this is not a mathematical generalization/superiority experiment.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from morph import MorphConfig
from morph.graph import ProofGraph,ProofNode
from morph.inference_payload import payload_prompt,autonomous_search
from morph.training.proof_runner import run_proof_training
from verifier import LeanClient


def graph_for(statement):
    graph=ProofGraph();graph.add(ProofNode(0,statement,formal_statement=statement));return graph


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--engine',choices=['resident','async','dls','disk'],default='resident')
    parser.add_argument('--state-dir',default='.generated/complete_proof_disk')
    parser.add_argument('--output',default='docs/proof_loop_results.json');args=parser.parse_args()
    torch.set_num_threads(2)
    client=LeanClient(project='.tools/mathlib4',imports=('Mathlib.Data.Nat.Basic',),timeout=10,log_path='.generated/proof_loop_receipts.jsonl')
    statements=['True','1 + 1 = 2','∀ n : Nat, n = n','∀ n : Nat, n + 0 = n','False -> True','True ∨ False']
    rows=[]
    def example(statement,proof,diagnostics='',trace_kind='verified_toy'):
        receipt=client.verify(statement,proof)
        if not receipt.success:raise RuntimeError(receipt.diagnostics)
        prompt=payload_prompt(statement,diagnostics=diagnostics)
        rows.append(dict(family=statement,split='train',text=prompt+json.dumps(dict(proof=proof))+'\0',payload_start=len(prompt.encode()),
                         graph=graph_for(statement).snapshot(),action=dict(type='VERIFY',target=0),value=1.,verification=asdict(receipt),trace_kind=trace_kind))
    for statement in statements:example(statement,'by simp')
    repair_goal='∀ n m : Nat, n + m = m + n'
    failed=client.verify(repair_goal,'by simp')
    if not failed.success:example(repair_goal,'by intro n m; omega',failed.diagnostics,'seeded_repair_training')
    root=Path('.generated');root.mkdir(exist_ok=True);path=root/'proof_loop_training.jsonl'
    path.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n')
    cfg=MorphConfig(vocab_size=256,d_model=32,num_layers=2,d_ff=64,d_state=4,head_dim=16,
                    d_graph=16,graph_nodes=2,graph_enabled=False,proof_enabled=True,statement_field_length=256)
    started=time.perf_counter()
    model,losses=run_proof_training(cfg,path,args.steps,lr=.005,sequence_length=1024,checkpoint=None if args.engine=='disk' else '.generated/proof_loop_model.pt',engine=args.engine,state_dir=args.state_dir)
    if hasattr(model,'eval'):model.eval()
    reports=[]
    for label,statement in [('heldout_toy','True ∧ True'),('trained_repair_fixture',repair_goal)]:
        result=autonomous_search(model,graph_for(statement),client,attempts=3,max_new_tokens=96)
        reports.append(dict(kind=label,statement=statement,**result))
    report=dict(training_engine=args.engine,training_steps=args.steps,training_loss_first=losses[0]['total'],training_loss_last=losses[-1]['total'],
                elapsed_seconds=time.perf_counter()-started,model_generated_payloads=True,runtime_tactic_fallback=False,
                seeded_training_failure=asdict(failed),searches=reports,
                claims='tiny supervised proof/repair fixture, not architecture superiority or novel theorem discovery')
    Path(args.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
