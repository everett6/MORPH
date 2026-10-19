"""Memory-mapped token windows and theorem-family-isolated proof datasets."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .tokenizer import ByteTokenizer


def compile_shard(records,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    manifest=dict(tokenizer=ByteTokenizer.fingerprint,families={},records=[],version=1)
    for index,record in enumerate(records):
        if not record.get('family'):
            raise ValueError('each proof record requires theorem family')
        split=record.get('split','train')
        prior=manifest['families'].setdefault(record['family'],split)
        if prior!=split:
            raise ValueError('theorem family leaks across splits')
        ids=np.asarray(ByteTokenizer().encode(record['text']),dtype=np.uint16)
        np.save(root/f'{index}.npy',ids)
        metadata={k:v for k,v in record.items() if k!='text'}
        metadata.update(file=f'{index}.npy',dtype='uint16',length=len(ids),sha256=hashlib.sha256(ids.tobytes()).hexdigest())
        manifest['records'].append(metadata)
    (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True))
    return root


class MappedShard:
    def __init__(self,root,split='train'):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest['tokenizer']!=ByteTokenizer.fingerprint:
            raise ValueError('tokenizer mismatch')
        if self.manifest.get('version')!=1:
            raise ValueError('unsupported shard version')
        families={}
        for record in self.manifest['records']:
            family,part=record['family'],record.get('split','train')
            if families.setdefault(family,part)!=part or self.manifest['families'].get(family)!=part:
                raise ValueError('theorem family split inconsistency')
        self.records=[r for r in self.manifest['records'] if r.get('split','train')==split]
        self.arrays=[]
        for record in self.records:
            path=(self.root/record['file']).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError('shard file escapes root')
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            if record.get('dtype')!='uint16' or array.dtype!=np.uint16 or array.ndim!=1 or len(array)!=record['length']:
                raise ValueError('token array schema mismatch')
            digest=hashlib.sha256()
            for start in range(0,len(array),1048576):
                chunk=array[start:start+1048576]
                if len(chunk) and chunk.max()>=256:
                    raise ValueError('token outside fixed vocabulary')
                digest.update(chunk.tobytes())
            if digest.hexdigest()!=record['sha256']:
                raise ValueError('token shard content hash mismatch')
            # Validation touches pages; reopen to discard that mapping's resident
            # working set before training retains only demand-paged windows.
            del array
            self.arrays.append(np.load(path,mmap_mode='r',allow_pickle=False))
        if not self.arrays:
            raise ValueError('no records in selected split')

    def window(self,index,start,length):
        array=self.arrays[index]
        if start<0 or start+length>len(array):
            raise ValueError('window outside record; theorem boundaries are preserved')
        return torch.from_numpy(np.asarray(array[start:start+length],dtype=np.int64).copy())

    def sample(self,generator,batch_size,length):
        eligible=[i for i,array in enumerate(self.arrays) if len(array)>=length]
        if not eligible:
            raise ValueError('no theorem record long enough')
        windows=[]
        for _ in range(batch_size):
            index=eligible[torch.randint(len(eligible),(1,),generator=generator).item()]
            start=torch.randint(len(self.arrays[index])-length+1,(1,),generator=generator).item()
            windows.append(self.window(index,start,length))
        return torch.stack(windows)
