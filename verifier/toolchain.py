"""Install pinned Lean/targeted Mathlib into ignored project-local storage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

LOCK=dict(lean_version='4.19.0',lean_commit='6caaee842e94',
          archive_url='https://github.com/leanprover/lean4/releases/download/v4.19.0/lean-4.19.0-linux.tar.zst',
          archive_sha256='6fe3ce97a58f44e2b3567d455b994eacec5bfe9ae7774f2a573444480ba813fe',
          mathlib_revision='c44e0c8ee63ca166450922a373c7409c5d26b00b',mathlib_tag='v4.19.0',
          imports=['Mathlib.Data.Nat.Basic','Mathlib.Logic.Basic'])


def install(root='.tools'):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    archive=root/'lean-4.19.0-linux.tar.zst';toolchain=root/'lean-4.19.0-linux'
    if not archive.exists():urllib.request.urlretrieve(LOCK['archive_url'],archive)
    digest=hashlib.sha256()
    with archive.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1048576),b''):digest.update(chunk)
    if digest.hexdigest()!=LOCK['archive_sha256']:raise RuntimeError('Lean archive checksum mismatch')
    if not (toolchain/'bin/lean').exists():subprocess.run(['tar','--zstd','-xf',str(archive),'-C',str(root)],check=True)
    project=root/'mathlib4'
    if not project.exists():subprocess.run(['git','clone','--depth','1','--branch',LOCK['mathlib_tag'],'https://github.com/leanprover-community/mathlib4.git',str(project)],check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    if revision!=LOCK['mathlib_revision']:raise RuntimeError('Mathlib revision mismatch')
    environment={**os.environ,'PATH':str(toolchain/'bin')+os.pathsep+os.environ.get('PATH','')}
    subprocess.run([str(toolchain/'bin/lake'),'exe','cache','get','Mathlib/Logic/Basic.lean','Mathlib/Data/Nat/Basic.lean'],cwd=project,env=environment,check=True)
    (root/'toolchain-lock.json').write_text(json.dumps(LOCK,indent=2)+'\n')
    return LOCK


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');parser.add_argument('--root',default='.tools')
    args=parser.parse_args();print(json.dumps(install(args.root) if args.install else LOCK,indent=2))


if __name__=='__main__':main()
