"""Compile only existing local Mathlib sources; never fetches data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .shards import compile_shard


def compile_local_math(source,output):
    source=Path(source).resolve();seen=set();counts={};total={}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source.parent,text=True).strip()
    files=sorted(source.rglob('*.lean'))
    def records():
        for path in files:
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            relative=path.relative_to(source)
            # Top-level mathematical module families stay together, including
            # all submodules. This is source/module isolation, not semantic dedup.
            family=relative.parts[0] if len(relative.parts)>1 else relative.stem
            bucket=int(hashlib.sha256(family.encode()).hexdigest(),16)%100
            split='test' if bucket<5 else 'validation' if bucket<10 else 'train'
            counts[split]=counts.get(split,0)+1;total[split]=total.get(split,0)+len(raw)
            yield dict(text=raw.decode('utf8'),family=family,split=split,source=str(relative),source_sha256=digest,
                       provenance=dict(repository='local Mathlib4',revision=revision,license='Apache-2.0'),trace_kind='lean_source')
    compile_shard(records(),output)
    report=dict(source=str(source),revision=revision,license='Apache-2.0',unique_files=len(seen),records=counts,byte_tokens=total,
                tokenizer='morph-byte-utf8-v1',downloaded=False,dedup='exact content SHA256',split='top-level module family hash; no semantic equivalence guarantee')
    Path(output,'corpus_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='.tools/mathlib4/Mathlib');p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(compile_local_math(a.source,a.output)))

if __name__=='__main__':main()
