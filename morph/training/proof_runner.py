"""Executable structured supervised training and constrained proof rollouts."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import torch
from morph import MorphConfig, MorphModel
from morph.graph import ActionType, ProofGraph, Status, apply_action
from morph.graph.tensors import tensorize
from .proof import joint_loss


def load_examples(path,split):
    examples=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    families={}
    for example in examples:
        family=example['family'];part=example['split']
        if families.setdefault(family,part)!=part:
            raise ValueError('theorem family appears in multiple splits')
    return [example for example in examples if example['split']==split]


def proof_batch(example,cfg,device):
    graph=ProofGraph.restore(example['graph'])
    tensors,maps=tensorize([graph],cfg.graph_nodes,device=device)
    action=example['action']
    action_id=ActionType[action['type']]
    target=maps[0][action['target']]
    tokens=torch.tensor([list(example['text'].encode())],device=device)
    if tokens.shape[1]<2:
        raise ValueError('proof text needs at least two bytes')
    return graph,tensors,tokens,torch.tensor([action_id],device=device),torch.tensor([target],device=device)


def run_proof_training(config,path,steps=10,lr=.001,device='cpu',checkpoint=None,split='train',sequence_length=64):
    if steps<0 or lr<=0 or sequence_length<2:
        raise ValueError('invalid proof training schedule')
    cfg=replace(config,proof_enabled=True)
    if cfg.vocab_size<256:
        raise ValueError('fixed proof byte tokenizer requires vocab_size>=256')
    examples=load_examples(path,split)
    if not examples:
        raise ValueError('selected split has no proof traces')
    torch.manual_seed(42)
    model=MorphModel(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    losses=[]
    for step in range(steps):
        example=examples[step%len(examples)]
        graph,tensors,tokens,actions,targets=proof_batch(example,cfg,device)
        tokens=tokens[:,:sequence_length]
        optimizer.zero_grad()
        outputs=model.forward_proof(tokens,tensors)
        loss,parts=joint_loss(outputs,tokens,tensors,actions,targets,torch.tensor([example['value']],device=device))
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite joint loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        # Actual teacher-forced discrete transition is performed outside autograd.
        event=apply_action(graph,example['action'])
        losses.append(dict(step=step+1,total=loss.item(),parts={k:v.item() for k,v in parts.items()},transition=event))
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(config=cfg.to_dict(),model=model.state_dict(),optimizer=optimizer.state_dict()),checkpoint)
    return model,losses


def rollout(model,text,graph,payload_provider,max_steps=8,device='cpu'):
    """Apply only valid action/target pairs; VERIFY records a request, never trust."""
    events=[]
    for _ in range(max_steps):
        if graph.stopped or not any(node.status not in (Status.BLOCKED,Status.REFUTED) for node in graph.nodes.values()):
            break
        tensors,maps=tensorize([graph],model.config.graph_nodes,device=device)
        tokens=torch.tensor([list(text.encode())],device=device)
        with torch.no_grad():
            outputs=model.forward_proof(tokens,tensors)
        action=int(outputs['action_logits'][0].argmax())
        target_index=int(outputs['target_logits'][0,action].argmax())
        inverse={index:node for node,index in maps[0].items()}
        if not tensors['target_action_mask'][0,action,target_index]:
            raise ValueError('controller selected invalid action/target pair')
        payload=payload_provider(ActionType(action),inverse[target_index],graph)
        if payload is None:
            break
        payload={**payload,'type':ActionType(action).name,'target':inverse[target_index]}
        proposed=copy.deepcopy(graph)
        apply_action(proposed,payload)
        if len(proposed.nodes)>model.config.graph_nodes:
            events.append(dict(rejected_action=payload,reason='graph capacity exceeded'))
            break
        events.append(apply_action(graph,payload))
        if graph.stopped:
            break
    return events
