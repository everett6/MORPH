"""Real Lean compiler boundary; no synthetic verifier success."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    statement: str
    assumptions: tuple[str, ...]
    proof_hash: str | None
    checker: str = 'lean4'
    version: str = 'unavailable'
    diagnostics: str = ''


def accepted_axiom_audit(output):
    lines=re.findall(r"^\s*'?morph_candidate'?\s+(does not depend on any axioms|depends on axioms:\s*\[[^\]]*\])\s*$",output,re.M)
    if len(lines)!=1:return False
    if lines[0]=='does not depend on any axioms':return True
    names=lines[0].split('[',1)[1].rsplit(']',1)[0]
    return all(name.strip() in {'propext','Classical.choice','Quot.sound'} for name in names.split(',') if name.strip())


class LeanClient:
    def __init__(self, project=None, timeout=10., log_path=None, imports=()):
        self.project=Path(project) if project else None
        self.timeout=timeout
        self.log_path=Path(log_path) if log_path else None
        self.imports=tuple(imports)
        if timeout<=0 or any(not re.fullmatch(r'[A-Za-z0-9_.]+',module) for module in imports):
            raise ValueError('invalid verifier configuration')
        self.command=(['lake','env','lean'] if self.project else ['lean'])

    @property
    def available(self):
        return shutil.which(self.command[0]) is not None

    def _run(self,args):
        process=subprocess.Popen(self.command+args,cwd=self.project,text=True,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        try:
            output,_=process.communicate(timeout=self.timeout)
            return process.returncode,output
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            output,_=process.communicate()
            return -1,'timeout\n'+output

    def verify(self,statement,proof,assumptions=()):
        assumptions=tuple(assumptions)
        forbidden=r'\b(sorry|admit|axiom|constant|unsafe|run_tac|set_option|theorem|def|opaque|namespace|import|end)\b|#|\x00'
        type_injection=any(any(separator in text for separator in ('\n','\r',':=')) for text in (statement,*assumptions))
        if type_injection or any(re.search(forbidden,text) for text in (statement,proof,*assumptions)):
            result=VerificationResult(False,statement,assumptions,None,diagnostics='forbidden declaration or trust bypass')
        elif not self.available:
            result=VerificationResult(False,statement,assumptions,None,diagnostics='Lean executable unavailable; verification not performed')
        else:
            source=''.join(f'import {module}\n' for module in self.imports)
            source+=f'theorem morph_candidate {" ".join(assumptions)} : {statement} := {proof}\n#print axioms morph_candidate\n'
            with tempfile.TemporaryDirectory(prefix='morph-lean-') as folder:
                path=Path(folder)/'Candidate.lean';path.write_text(source)
                code,output=self._run([str(path)])
                _,version=self._run(['--version'])
            # Closed theorem must compile and its axiom audit must be present.
            success=code==0 and accepted_axiom_audit(output)
            result=VerificationResult(success,statement,assumptions,hashlib.sha256(source.encode()).hexdigest() if success else None,
                                      version=version.strip(),diagnostics=output[-8000:])
        if self.log_path:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as handle:
                handle.write(json.dumps(asdict(result))+'\n')
        return result
