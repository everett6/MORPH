"""Bounded, durable stage undo log; payload volume equals legacy journals.

Only one 1 MiB copy buffer is used. The complete index and payload are validated
before recovery writes any canonical file. Cursor publication uses DiskState.
"""
import hashlib
import json
import os
import shutil
from .disk import DiskState, durable_json, profile_fsync


class PackedStateMixin:
    journal_format='packed-v1'

    @classmethod
    def create(cls,*args,**kwargs):
        store=super().create(*args,**kwargs)
        store.manifest['journal_format']=cls.journal_format
        durable_json(store.root/'manifest.json',store.manifest)
        return store

    @classmethod
    def open(cls,root,model,**kwargs):
        from pathlib import Path
        manifest=json.loads((Path(root)/'manifest.json').read_text())
        if manifest.get('journal_format')!=cls.journal_format:
            raise ValueError('disk journal format differs; packed state required')
        # CachedDiskState.open calls DiskState.open with the legacy class. Use
        # the base classmethod bound to this format-aware class instead.
        store=DiskState.open.__func__(cls,root,model)
        if kwargs:
            if set(kwargs)!={'cache_bytes'} or not hasattr(store,'configure_cache'):raise TypeError('unexpected open options')
            store.configure_cache(kwargs['cache_bytes'])
        return store

    def prepare_stage(self,index,names):
        names=list(names)
        if len(names)!=len(set(names)):raise ValueError('duplicate journal tensor')
        journal=self.root/'journal'
        shutil.rmtree(journal,ignore_errors=True);journal.mkdir()
        records=[];offset=0
        with (journal/'state.bin').open('wb') as destination:
            for name in names:
                entry=self.entries[name];size=entry['size']*12;digest=hashlib.sha256()
                with (self.root/entry['file']).open('rb') as source:
                    if os.fstat(source.fileno()).st_size!=size:raise ValueError('canonical state size differs')
                    while block:=source.read(1024*1024):
                        destination.write(block);digest.update(block)
                records.append(dict(name=name,file=entry['file'],offset=offset,size=size,sha256=digest.hexdigest()))
                offset+=size
            destination.flush();profile_fsync(destination.fileno())
        durable_json(journal/'index.json',dict(format=self.journal_format,stage=index,bytes=offset,records=records))
        descriptor=os.open(self.root,os.O_RDONLY)
        try:profile_fsync(descriptor)
        finally:os.close(descriptor)
        self.fault('journal_durable')
        self.manifest['transaction']['journal_hash']=hashlib.sha256((journal/'index.json').read_bytes()).hexdigest()
        self.manifest['transaction']['journal']=index
        durable_json(self.root/'manifest.json',self.manifest)

    def recover_stage(self):
        transaction=self.manifest.get('transaction')
        if not transaction:return
        journal=self.root/'journal'
        if transaction.get('journal') is not None:
            try:
                raw=(journal/'index.json').read_bytes()
                if hashlib.sha256(raw).hexdigest()!=transaction.get('journal_hash'):raise ValueError('index checksum differs')
                index=json.loads(raw)
                if set(index)!={'format','stage','bytes','records'} or index['format']!=self.journal_format or index['stage']!=transaction['journal']:raise ValueError('invalid index')
                offset=0;seen=set()
                with (journal/'state.bin').open('rb') as source:
                    for record in index['records']:
                        if set(record)!={'name','file','offset','size','sha256'}:raise ValueError('invalid record')
                        name=record['name'];entry=self.entries[name]
                        if name in seen or record['file']!=entry['file'] or record['offset']!=offset or record['size']!=entry['size']*12:raise ValueError('invalid schema')
                        seen.add(name);remaining=record['size'];digest=hashlib.sha256()
                        while remaining:
                            block=source.read(min(remaining,1024*1024))
                            if not block:raise ValueError('truncated payload')
                            digest.update(block);remaining-=len(block)
                        if digest.hexdigest()!=record['sha256']:raise ValueError('checksum differs')
                        offset+=record['size']
                    if index['bytes']!=offset or os.fstat(source.fileno()).st_size!=offset:raise ValueError('payload size differs')
                for record in index['records']:
                    if (self.root/record['file']).stat().st_size!=record['size']:raise ValueError('canonical state size differs')
                # Validate the complete journal before touching any canonical file.
                with (journal/'state.bin').open('rb') as source:
                    for record in index['records']:
                        with (self.root/record['file']).open('r+b') as destination:
                            remaining=record['size']
                            while remaining:
                                block=source.read(min(remaining,1024*1024));destination.write(block);remaining-=len(block)
                            destination.flush();profile_fsync(destination.fileno())
            except (OSError,KeyError,TypeError,ValueError) as error:
                raise ValueError(f'invalid packed journal: {error}') from error
            transaction['journal']=None
            durable_json(self.root/'manifest.json',self.manifest)
        shutil.rmtree(journal,ignore_errors=True)
        for path in self.root.glob('cursor-*.pt'):
            if path.name!=transaction['cursor']:path.unlink()


class PackedDiskState(PackedStateMixin,DiskState):
    pass

from .state_cache import CachedDiskState

class PackedCachedDiskState(PackedStateMixin,CachedDiskState):
    def recover_stage(self):
        with self._cache_lock:
            transaction=self.manifest.get('transaction')
            if transaction is not None and transaction.get('journal') is not None:
                self._cache.clear();self._cache_bytes=0
            return super().recover_stage()
