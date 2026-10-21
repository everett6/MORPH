"""Bounded FP16 model export from clean FP32 canonical disk state."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from morph import MorphConfig,MorphModel
from morph.factory import config_from_dict,model_from_config
from .disk import DiskState,durable_json


def export_fp16(state_dir,output,chunk_elements=1048576,lock_held=False):
    state_dir=Path(state_dir);output=Path(output)
    with nullcontext(None) if lock_held else (state_dir.parent/'run.lock').open('a+') as lock:
        if lock is not None:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('stop training safely before exporting a consistent model')
        manifest=json.loads((state_dir/'manifest.json').read_text())
        if manifest['dirty']:raise ValueError('cannot export an incomplete optimizer transaction')
        with torch.device('meta'):model=model_from_config(config_from_dict(manifest['config']))
        journal_format=manifest.get('journal_format','legacy')
        if journal_format=='legacy':state_class=DiskState
        elif journal_format=='packed-v1':
            from .packed_state import PackedDiskState
            state_class=PackedDiskState
        else:raise ValueError('unsupported disk journal format')
        store=state_class.open(state_dir,model)
        output.mkdir(parents=True,exist_ok=False);weights={}
        for name,entry in store.entries.items():
            source=np.memmap(state_dir/entry['file'],dtype=np.float32,mode='r',shape=(3,entry['size']))
            path=output/entry['file'];digest=hashlib.sha256()
            with path.open('wb') as handle:
                for start in range(0,entry['size'],chunk_elements):
                    value=np.asarray(source[0,start:start+chunk_elements])
                    if not np.isfinite(value).all() or np.abs(value).max(initial=0)>65504:raise ValueError('weights cannot be represented by finite FP16')
                    encoded=value.astype('<f2').tobytes();handle.write(encoded);digest.update(encoded)
                handle.flush();os.fsync(handle.fileno())
            del source
            weights[name]=dict(file=path.name,shape=entry['shape'],dtype='float16',sha256=digest.hexdigest())
        canonical={id(parameter):name for name,parameter in model.named_parameters()}
        aliases={name:canonical[id(parameter)] for name,parameter in model.named_parameters(remove_duplicate=False) if canonical[id(parameter)]!=name}
        result=dict(format='morph-fp16-shards-v1',config=model.config.to_dict(),step=manifest['step'],weights=weights,
                    aliases=aliases,lm_head=dict(kind='tied_embedding_projection',weight='embedding.weight'),parameters=sum(e['size'] for e in store.entries.values()),optimizer_included=False,
                    buffers='deterministically reconstructed from config',source_precision='float32 masters',export_precision='float16')
        durable_json(output/'manifest.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=export_fp16(a.state_dir,a.output);print(json.dumps(dict(step=result['step'],parameters=result['parameters'],output=a.output)))

if __name__=='__main__':main()


def load_fp16(output,device='cpu'):
    """Validate and map FP16 shards directly into a meta-defined inference model.

    CPU tensors use private copy-on-write mappings; no FP32 master allocation.
    CUDA loading copies those FP16 views only after a free-memory guard.
    """
    output=Path(output).resolve();manifest=json.loads((output/'manifest.json').read_text())
    if manifest.get('format')!='morph-fp16-shards-v1':raise ValueError('unsupported export format')
    with torch.device('meta'):model=model_from_config(config_from_dict(manifest['config']))
    expected={name:(list(p.shape),p.numel()) for name,p in model.named_parameters()}
    if set(expected)!=set(manifest['weights']):raise ValueError('export parameter schema differs')
    if manifest.get('lm_head')!={'kind':'tied_embedding_projection','weight':'embedding.weight'}:
        raise ValueError('export LM output tying metadata differs')
    count=sum(size for _,size in expected.values())
    if count!=manifest['parameters']:raise ValueError('export parameter count differs')
    canonical={id(p):name for name,p in model.named_parameters()}
    aliases={name:canonical[id(p)] for name,p in model.named_parameters(remove_duplicate=False) if canonical[id(p)]!=name}
    if aliases!=manifest['aliases']:raise ValueError('export tied-parameter aliases differ')
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available() or count*2+512*2**20>torch.cuda.mem_get_info(device)[0]*.8:
            raise MemoryError('FP16 weights plus inference headroom exceed free CUDA memory; use CPU mapped model')
    parameters={};mappings=[]
    for name,(shape,size) in expected.items():
        entry=manifest['weights'][name];path=(output/entry['file']).resolve()
        if not path.is_relative_to(output) or entry['dtype']!='float16' or entry['shape']!=shape or path.stat().st_size!=size*2:
            raise ValueError('invalid FP16 shard schema/path/length')
        mapped=np.memmap(path,dtype='<f2',mode='c',shape=tuple(shape));digest=hashlib.sha256()
        for start in range(0,size,1048576):
            chunk=mapped.reshape(-1)[start:start+1048576]
            if not np.isfinite(chunk).all():raise ValueError('nonfinite FP16 exported weights')
            digest.update(chunk.tobytes())
        if digest.hexdigest()!=entry['sha256']:raise ValueError('FP16 export hash mismatch')
        parameters[name]=torch.nn.Parameter(torch.from_numpy(mapped),requires_grad=False);mappings.append(mapped)
    for name,canonical_name in {**{name:name for name in expected},**aliases}.items():
        parent,_,leaf=name.rpartition('.');module=model.get_submodule(parent) if parent else model
        module._parameters[leaf]=parameters[canonical_name]
    for module in model.modules():
        if hasattr(module,'sources'):
            capacity=module.sources.shape[1]
            module.sources=torch.arange(capacity).view(1,-1,1)
            module.targets=torch.arange(capacity).roll(-1).view(1,-1)
    model._fp16_mappings=mappings
    model.eval()
    return model.to(device) if str(device)!='cpu' else model
