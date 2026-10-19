"""Evaluate supervised proof heads on held-out families; no proof-validity proxy."""
import argparse
import json
from pathlib import Path
import torch
from morph import MorphConfig,MorphModel
from morph.training.proof_runner import load_examples,proof_batch
from morph.training.proof import joint_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--proof-disk-state')
    parser.add_argument('--search-statement')
    parser.add_argument('--assumption',action='append',default=[])
    parser.add_argument('--lean-project')
    parser.add_argument('--search-policy',choices=['learned','verify'],default='learned')
    parser.add_argument('--attempts',type=int,default=3)
    parser.add_argument('--proof-traces')
    parser.add_argument('--split',default='validation')
    args=parser.parse_args()
    torch.set_num_threads(2)
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    if args.proof_disk_state:
        from streamtrain.disk import DiskState
        from streamtrain.proof_inference import DiskProofModel
        manifest=json.loads((Path(args.proof_disk_state)/'manifest.json').read_text())
        with torch.device('meta'):template=MorphModel(MorphConfig(**manifest['config']))
        from streamtrain.disk import estimate_disk_budget
        budget=estimate_disk_budget(template,1,256,1,torch.float32)
        if budget['estimated_host_bytes']+budget['estimated_cuda_bytes']>available*.75:
            parser.error('disk inference stage/workspace estimate exceeds available RAM')
        store=DiskState.open(args.proof_disk_state,template);model=DiskProofModel(template,store)
    else:
        if not args.checkpoint:parser.error('provide --checkpoint or --proof-disk-state')
        if Path(args.checkpoint).stat().st_size*3+512*2**20>available*.75:
            parser.error('checkpoint load/model duplication exceeds available RAM; use disk inference')
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model=MorphModel(MorphConfig(**saved['config']));model.load_state_dict(saved['model']);model.eval()
    if args.search_statement:
        from morph.graph import ProofGraph,ProofNode,obligation_closed
        from morph.training.proof_runner import rollout
        from morph.inference_payload import autonomous_search
        from verifier import LeanClient
        graph=ProofGraph();graph.add(ProofNode(0,args.search_statement,formal_statement=args.search_statement,assumptions=tuple(args.assumption)))
        verifier=LeanClient(project=args.lean_project,imports=('Mathlib.Data.Nat.Basic',) if args.lean_project else ())
        if args.search_policy=='verify':result=autonomous_search(model,graph,verifier,attempts=args.attempts)
        else:
            events=rollout(model,'Goal: '+args.search_statement,graph,max_steps=args.attempts,verifier=verifier)
            result=dict(events=events,graph=graph.snapshot(),success=(graph.nodes[0].formal_statement==args.search_statement and graph.nodes[0].assumptions==tuple(args.assumption) and obligation_closed(graph,0)),action_policy='learned')
        print(json.dumps(result));return
    if not args.proof_traces:parser.error('provide --proof-traces or --search-statement')
    results=[]
    for example in load_examples(args.proof_traces,args.split):
        graph,tensors,tokens,actions,targets=proof_batch(example,model.config,'cpu')
        with torch.no_grad():
            output=model.forward_proof(tokens,tensors)
            loss,_=joint_loss(output,tokens,tensors,actions,targets,torch.tensor([example['value']]))
        results.append(dict(family=example['family'],loss=loss.item(),action_correct=int(output['action_logits'].argmax(-1)==actions)))
    if not results:parser.error('selected validation split has no examples')
    print(json.dumps(dict(examples=results,verified_proofs=None,verification='not performed')))


if __name__=='__main__':main()
