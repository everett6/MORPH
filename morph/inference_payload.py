"""Model-byte payload decoding and a verifier-driven autonomous search loop."""
import json
import time
from dataclasses import asdict
import torch
from morph.graph import ActionType,ProofNode,Status,apply_action,obligation_closed
from morph.graph.tensors import tensorize


def payload_prompt(statement,kind='VERIFY',diagnostics=''):
    return f'Goal: {statement}\nAction: {kind}\nDiagnostics: {diagnostics[-1024:]}\nPayload: '


class PayloadDecoder:
    def __init__(self,model,max_new_tokens=256,device='cpu'):
        self.model,self.max_new_tokens,self.device=model,max_new_tokens,device

    def decode(self,graph,target,kind,diagnostics=''):
        node=graph.nodes[target]
        prompt=payload_prompt(node.formal_statement or node.statement,kind.name,diagnostics)
        tensors,_=tensorize([graph],self.model.config.graph_nodes,self.model.config.statement_field_length,self.device)
        ids=torch.tensor([list(prompt.encode())],device=self.device)
        states=None;generated=[]
        for step in range(self.max_new_tokens):
            with torch.no_grad():
                output=self.model.forward_proof(ids if step==0 or not getattr(self.model,'supports_cache',True) else ids[:,-1:],tensors,states)
            states=output['states'];logits=output['logits'][0,-1].float()
            # Fixed byte grammar only; no hand-authored tactic or proof fallback.
            allowed=torch.zeros_like(logits,dtype=torch.bool);allowed[0]=True;allowed[9:11]=True;allowed[32:127]=True
            token=int(logits.masked_fill(~allowed,-torch.inf).argmax())
            if token==0:break
            generated.append(token);ids=torch.cat((ids,torch.tensor([[token]],device=self.device)),1)
            # Stop on a complete JSON object, including quoted braces/escapes.
            try:
                candidate=json.loads(bytes(generated).decode('utf8'))
                if isinstance(candidate,dict):break
            except (ValueError,UnicodeDecodeError):pass
        raw=bytes(generated).decode('utf8',errors='replace')
        try:
            value=json.loads(raw)
            if not isinstance(value,dict):raise ValueError('payload must be an object')
        except (ValueError,json.JSONDecodeError) as error:
            return dict(valid=False,raw=raw,error=str(error),model_generated=True,generated_tokens=len(generated))
        try:validate_payload(value,kind)
        except ValueError as error:return dict(valid=False,raw=raw,error=str(error),model_generated=True,generated_tokens=len(generated))
        return dict(valid=True,payload=value,raw=raw,model_generated=True,generated_tokens=len(generated))


def validate_payload(value,kind):
    fields={ActionType.VERIFY:{'proof'},ActionType.TEST:{'request'},
            ActionType.CREATE_LEMMA:{'statement','formal_statement','assumptions'},
            ActionType.DECOMPOSE:{'statement','formal_statement','assumptions'},
            ActionType.RETRIEVE:{'statement','formal_statement','assumptions'},
            ActionType.REVISE:{'statement','formal_statement','assumptions'},
            ActionType.CONNECT:{'edge'},ActionType.MERGE:{'edge'}}
    allowed=fields.get(kind,set())
    if set(value)-allowed:raise ValueError('unknown or control-overriding payload fields')
    required='proof' if kind==ActionType.VERIFY else 'request' if kind==ActionType.TEST else 'statement' if kind in (ActionType.CREATE_LEMMA,ActionType.DECOMPOSE,ActionType.RETRIEVE,ActionType.REVISE) else None
    if required and (not isinstance(value.get(required),str) or not value[required].strip()):raise ValueError(required+' string required')
    if 'assumptions' in value and (not isinstance(value['assumptions'],list) or not all(isinstance(x,str) for x in value['assumptions'])):raise ValueError('assumptions must be explicit strings')
    if 'formal_statement' in value and value['formal_statement'] is not None and not isinstance(value['formal_statement'],str):raise ValueError('formal statement must be a string')
    if kind in (ActionType.CONNECT,ActionType.MERGE) and not isinstance(value.get('edge'),dict):raise ValueError('typed edge required')


def autonomous_search(model,graph,verifier,attempts=4,target=0,max_new_tokens=256,device='cpu'):
    decoder=PayloadDecoder(model,max_new_tokens,device)
    diagnostics='';events=[]
    for attempt in range(attempts):
        started=time.perf_counter()
        decoded=decoder.decode(graph,target,ActionType.VERIFY,diagnostics)
        event=dict(attempt=attempt,decoded=decoded,decode_seconds=time.perf_counter()-started,diagnostics_context=diagnostics,prompt_graph=graph.snapshot())
        if not decoded['valid']:
            diagnostics=decoded['error'];events.append(event);continue
        proof=decoded['payload']['proof'];node=graph.nodes[target]
        started=time.perf_counter()
        result=verifier.verify(node.formal_statement,proof,node.assumptions)
        event['compiler_seconds']=time.perf_counter()-started
        event['verification']=asdict(result);events.append(event)
        if result.success:
            graph.accept_verification(target,result)
            return dict(success=obligation_closed(graph,target),events=events,graph=graph.snapshot(),action_policy='fixed VERIFY schedule')
        diagnostics=result.diagnostics
    return dict(success=False,events=events,graph=graph.snapshot(),action_policy='fixed VERIFY schedule')


def apply_generated_payload(graph,target,kind,decoded):
    if not decoded['valid']:raise ValueError(decoded['error'])
    data=decoded['payload'];validate_payload(data,kind)
    if kind in (ActionType.CREATE_LEMMA,ActionType.DECOMPOSE,ActionType.RETRIEVE):
        new_id=max(graph.nodes,default=-1)+1
        action=dict(type=kind.name,target=target,nodes=[dict(node_id=new_id,statement=data['statement'],
                    formal_statement=data.get('formal_statement'),assumptions=data.get('assumptions',list(graph.nodes[target].assumptions)))])
    elif kind==ActionType.REVISE:action=dict(type=kind.name,target=target,**data)
    elif kind==ActionType.TEST:return dict(request='counterexample',target=target,**data)
    else:action=dict(type=kind.name,target=target,**data)
    return apply_action(graph,action)
