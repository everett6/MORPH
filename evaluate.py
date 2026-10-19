"""Evaluate supervised proof heads on held-out families; no proof-validity proxy."""
import argparse
import json
import torch
from morph import MorphConfig,MorphModel
from morph.training.proof_runner import load_examples,proof_batch
from morph.training.proof import joint_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--proof-traces',required=True)
    parser.add_argument('--split',default='validation')
    args=parser.parse_args()
    torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=MorphModel(MorphConfig(**saved['config']));model.load_state_dict(saved['model']);model.eval()
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
