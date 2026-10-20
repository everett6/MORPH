import hashlib
import json
from pathlib import Path
from data.local_math import compile_local_math
from data import MappedShard


def test_local_compiler_deduplicates_and_splits_module_families(tmp_path,monkeypatch):
    source=tmp_path/'Mathlib';source.mkdir()
    families={}
    for index in range(100):
        name=f'Family{index}';bucket=int(hashlib.sha256(name.encode()).hexdigest(),16)%100
        split='test' if bucket<5 else 'validation' if bucket<10 else 'train'
        families.setdefault(split,name)
    for split,name in families.items():
        folder=source/name;folder.mkdir()
        (folder/'First.lean').write_text(f'-- {name}\ntheorem first : True := by trivial\n')
        (folder/'Second.lean').write_text(f'-- {name}\ntheorem second : True := by trivial\n')
    duplicate=source/'ZZZDuplicate.lean';duplicate.write_bytes(next(source.rglob('First.lean')).read_bytes())
    monkeypatch.setattr('data.local_math.subprocess.check_output',lambda *a,**k:'fixture-local-revision\n')
    output=tmp_path/'mapped';report=compile_local_math(source,output)
    assert report['unique_files']==6 and report['downloaded'] is False
    manifest=json.loads((output/'manifest.json').read_text())
    for split in ('train','validation','test'):
        shard=MappedShard(output,split)
        assert len(shard.records)==2
        assert {record['family'] for record in shard.records}=={families[split]}
    assert all(record['provenance']['license']=='Apache-2.0' for record in manifest['records'])
