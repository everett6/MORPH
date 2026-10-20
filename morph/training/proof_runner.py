"""Executable structured supervised training and constrained proof rollouts."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import torch
from morph.factory import config_from_dict, model_from_config
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
    tensors,maps=tensorize([graph],cfg.graph_nodes,cfg.statement_field_length,device=device)
    action=example['action']
    action_id=ActionType[action['type']]
    target=maps[0][action['target']]
    tokens=torch.tensor([list(example['text'].encode())],device=device)
    if tokens.shape[1]<2:
        raise ValueError('proof text needs at least two bytes')
    return graph,tensors,tokens,torch.tensor([action_id],device=device),torch.tensor([target],device=device)


def collate_proof(examples,cfg,sequence_length,device='cpu'):
    graphs=[ProofGraph.restore(example['graph']) for example in examples]
    tensors,maps=tensorize(graphs,cfg.graph_nodes,cfg.statement_field_length,device=device)
    if any(len(example['text'].encode())>sequence_length and (getattr(cfg,'model_type',None)=='delta' or 'payload_start' in example) for example in examples):
        raise ValueError('proof sequence would truncate its text/payload; increase sequence_length')
    texts=[list(example['text'].encode())[:sequence_length] for example in examples]
    if any(len(text)<2 for text in texts):raise ValueError('proof text needs at least two tokens')
    length=max(map(len,texts))
    tokens=torch.zeros(len(texts),length,dtype=torch.long,device=device)
    mask=torch.zeros_like(tokens,dtype=torch.bool)
    for index,text in enumerate(texts):
        tokens[index,:len(text)]=torch.tensor(text,device=device);mask[index,:len(text)]=True
    tensors['text_mask']=mask
    loss_mask=mask.clone()
    for index,example in enumerate(examples):
        if 'payload_start' in example:loss_mask[index,:min(example['payload_start'],length)]=False
    tensors['loss_mask']=loss_mask
    value_only=[example.get('value_only',False) for example in examples]
    if any(value_only) and not all(value_only):raise ValueError('do not mix value-only and joint examples in one batch')
    output=dict(tokens=tokens,graph=tensors,
                actions=torch.tensor([ActionType[example['action']['type']] for example in examples],device=device),
                targets=torch.tensor([mapping[example['action']['target']] for example,mapping in zip(examples,maps)],device=device),
                values=torch.tensor([example['value'] for example in examples],device=device))
    if all(value_only):output['weights']=dict(token=0.,graph=0.,action=0.,target=0.,value=1.)
    return output


def run_proof_training(config,path,steps=10,lr=.001,device='cpu',checkpoint=None,split='train',sequence_length=64,
                       engine='resident',batch_size=1,accumulation=1,dtype=torch.float32,state_dir=None,resume=False,checkpoint_every=100,resume_path=None,seed=42,continue_lm_state=False):
    if steps<0 or lr<=0 or sequence_length<2 or batch_size<1 or accumulation<1:
        raise ValueError('invalid proof training schedule')
    if dtype==torch.float16 and engine!='disk':
        raise ValueError('FP16 proof training requires disk engine with dynamic loss scaling')
    cfg=replace(config,proof_enabled=True)
    if cfg.vocab_size<256:raise ValueError('fixed proof byte tokenizer requires vocab_size>=256')
    examples=load_examples(path,split)
    if not examples:raise ValueError('selected split has no proof traces')
    import hashlib
    from streamtrain.disk import durable_json,durable_torch
    settings=dict(seed=seed,fingerprint=hashlib.sha256(Path(path).read_bytes()).hexdigest(),lr=lr,engine=engine,device_type=torch.device(device).type,
                  batch_size=batch_size,accumulation=accumulation,dtype=str(dtype),sequence_length=sequence_length,split=split)
    torch.manual_seed(seed)
    from streamtrain import StreamTrainEngine
    from streamtrain.disk import DiskDLS,DiskState
    if engine=='disk':
        if state_dir is None:raise ValueError('disk proof training requires state_dir')
        with torch.device('meta'):model=model_from_config(cfg)
        store=DiskState.open(state_dir,model) if resume else DiskState.create(state_dir,model,seed=seed)
        if resume and store.manifest.get('proof_settings')!=settings:
            if not continue_lm_state or store.manifest.get('proof_settings') is not None:
                raise ValueError('proof resume data/settings differ')
            if store.manifest.get('dirty') or store.manifest.get('transaction'):
                raise ValueError('LM-to-proof transition requires a clean state')
            progress_path=store.root.parent/'stream-state.json'
            previous=None
            if progress_path.exists():
                previous=json.loads(progress_path.read_text())
                if previous.get('source',{}).get('pending') is not None:raise ValueError('commit pending LM source before transition')
            store.manifest['phase_transition']=dict(from_phase='language_model',to_phase='joint_proof',step=store.manifest['step'],
                previous_settings=store.manifest.get('settings'),previous_proof_settings=None,previous_source_progress=previous)
        store.manifest['proof_settings']=settings;durable_json(store.root/'manifest.json',store.manifest)
        from streamtrain.sparse_disk import SparseDiskDLS
        trainer_type=SparseDiskDLS if getattr(cfg,'model_type',None)=='delta' else DiskDLS
        trainer=trainer_type(model,store,device,dtype,lr,async_mode=str(device).startswith('cuda'),cpu_prefetch=True)
        if store.manifest.get('transaction'):trainer.resume_step()
        start=store.manifest['step'];optimizer=None
    else:
        model=model_from_config(cfg).to(device if engine=='resident' else 'cpu')
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01,foreach=False)
        trainer=None if engine=='resident' else StreamTrainEngine(model,device,engine=='async',dtype)
        start=0
        if resume:
            saved=torch.load(resume_path or checkpoint,map_location='cpu',weights_only=True)
            if config_from_dict(saved['config']).to_dict()!=cfg.to_dict() or saved['settings']!=settings:raise ValueError('proof resume config/data/settings differ')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
            torch.set_rng_state(saved['rng'])
            if device.startswith('cuda'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if steps<start:raise ValueError('total steps precedes stored progress')
    def save_proof(step):
        if checkpoint and engine!='disk':
            Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
            durable_torch(Path(checkpoint),dict(config=cfg.to_dict(),settings=settings,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                          rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []))
    losses=[]
    for step in range(start,steps):
        microbatches=[]
        for micro in range(accumulation):
            selected=[examples[(step*accumulation*batch_size+micro*batch_size+i)%len(examples)] for i in range(batch_size)]
            microbatches.append(collate_proof(selected,cfg,sequence_length))
        if engine=='disk':
            result=trainer.step(microbatches);total=result['loss']
        else:
            optimizer.zero_grad();total=0.
            for batch in microbatches:
                if trainer is not None:
                    total+=trainer.backward(batch,1/accumulation)/accumulation
                else:
                    tokens=batch['tokens'].to(device);graph={k:v.to(device) for k,v in batch['graph'].items()}
                    with torch.autocast(torch.device(device).type,dtype=dtype,enabled=dtype==torch.bfloat16):
                        output=model.forward_proof(tokens,graph)
                        loss,_=joint_loss(output,tokens,graph,batch['actions'].to(device),batch['targets'].to(device),batch['values'].to(device),batch.get('weights'))
                        loss=loss+output.get('router_aux_loss',0)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite joint loss')
                    (loss/accumulation).backward();total+=loss.item()/accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if trainer is not None:trainer.parameter_version+=1
        losses.append(dict(step=step+1,total=total))
        if checkpoint_every and (step+1)%checkpoint_every==0:save_proof(step+1)
    save_proof(steps)
    if engine=='disk':
        from streamtrain.proof_inference import DiskProofModel
        return DiskProofModel(model,store,device,dtype,executor=trainer),losses
    return model,losses


def rollout(model,text,graph,payload_provider=None,max_steps=8,device='cpu',verifier=None):
    """Apply only valid action/target pairs; VERIFY records a request, never trust."""
    events=[];diagnostics=''
    from morph.graph import NodeType
    anchors={index:(node.statement,node.formal_statement,node.assumptions) for index,node in graph.nodes.items() if node.node_type==NodeType.GOAL}
    def check_anchors():
        if any(index not in graph.nodes or (graph.nodes[index].statement,graph.nodes[index].formal_statement,graph.nodes[index].assumptions)!=identity for index,identity in anchors.items()):
            raise ValueError('original proof goal changed during search')
    if payload_provider is None:
        from morph.inference_payload import PayloadDecoder,apply_generated_payload
        decoder=PayloadDecoder(model,device=device)
    for _ in range(max_steps):
        check_anchors()
        if graph.stopped or not any(node.status not in (Status.BLOCKED,Status.REFUTED) for node in graph.nodes.values()):
            break
        tensors,maps=tensorize([graph],model.config.graph_nodes,model.config.statement_field_length,device=device)
        tokens=torch.tensor([list(text.encode())],device=device)
        with torch.no_grad():
            outputs=model.forward_proof(tokens,tensors)
        action=int(outputs['action_logits'][0].argmax())
        target_index=int(outputs['target_logits'][0,action].argmax())
        inverse={index:node for node,index in maps[0].items()}
        if not tensors['target_action_mask'][0,action,target_index]:
            raise ValueError('controller selected invalid action/target pair')
        if payload_provider is None:
            target=inverse[target_index];kind=ActionType(action)
            if kind==ActionType.STOP:
                events.append(apply_action(graph,dict(type='STOP',target=target)));break
            decoded=decoder.decode(graph,target,kind,diagnostics)
            if not decoded['valid']:
                diagnostics=decoded['error'];events.append(dict(rejected_payload=decoded));continue
            if kind==ActionType.VERIFY:
                event=dict(action='VERIFY',target=target,decoded=decoded)
                if verifier is None:
                    event['status']='verification_requested';events.append(event);break
                from dataclasses import asdict
                node=graph.nodes[target]
                result=verifier.verify(node.formal_statement,decoded['payload']['proof'],node.assumptions)
                event['verification']=asdict(result);events.append(event)
                if result.success:
                    graph.accept_verification(target,result)
                    from morph.graph import obligation_closed
                    if obligation_closed(graph,target):break
                diagnostics=result.diagnostics
                continue
            proposed=copy.deepcopy(graph)
            event=apply_generated_payload(proposed,target,kind,decoded)
            if len(proposed.nodes)>model.config.graph_nodes:
                events.append(dict(rejected_action=decoded,reason='graph capacity exceeded'));break
            apply_generated_payload(graph,target,kind,decoded);events.append(dict(event=event,decoded=decoded))
            continue
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
    check_anchors()
    return events
