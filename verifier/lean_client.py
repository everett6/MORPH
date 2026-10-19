"""Real Lean compiler boundary; no synthetic verifier success."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import resource
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
    isolated: bool = False
    project_revision: str | None = None


def accepted_axiom_audit(output):
    lines=re.findall(r"^\s*'?morph_candidate'?\s+(does not depend on any axioms|depends on axioms:\s*\[[^\]]*\])\s*$",output,re.M)
    if len(lines)!=1:return False
    if lines[0]=='does not depend on any axioms':return True
    names=lines[0].split('[',1)[1].rsplit(']',1)[0]
    return all(name.strip() in {'propext','Classical.choice','Quot.sound'} for name in names.split(',') if name.strip())


class LeanClient:
    def __init__(self, project=None, timeout=10., log_path=None, imports=(), isolated=True, executable=None):
        self.project=Path(project).resolve() if project else None
        local=Path(__file__).resolve().parents[1]/'.tools/lean-4.19.0-linux/bin'
        self.bin_dir=local if (local/'lean').exists() else None
        self.isolated=isolated
        self.project_revision=None
        if self.project and (self.project/'.git').exists():
            self.project_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.project,text=True).strip()
        self.timeout=timeout
        self.log_path=Path(log_path) if log_path else None
        self.imports=tuple(imports)
        if timeout<=0 or any(not re.fullmatch(r'[A-Za-z0-9_.]+',module) for module in imports):
            raise ValueError('invalid verifier configuration')
        lean=executable or str(self.bin_dir/'lean') if self.bin_dir else executable or 'lean'
        lake=str(self.bin_dir/'lake') if self.bin_dir else 'lake'
        self.command=([lake,'env',lean] if self.project else [lean])

    @property
    def available(self):
        return shutil.which(self.command[0]) is not None

    def _run(self,args):
        command=self.command+args
        environment=dict(PATH=f'{self.bin_dir}:/usr/bin:/bin' if self.bin_dir else '/usr/bin:/bin',HOME='/tmp',LANG='C.UTF-8')
        if self.isolated:
            if not shutil.which('bwrap'):return -1,'rootless bubblewrap isolation unavailable'
            wrapper=['bwrap','--die-with-parent','--new-session','--unshare-all']
            for path in ['/usr','/bin','/lib','/lib64']:
                if Path(path).exists():wrapper+=['--ro-bind',path,path]
            if self.bin_dir:
                root=str(self.bin_dir.parent);wrapper+=['--ro-bind',root,root]
            elif '/' in self.command[0]:
                root=str(Path(self.command[0]).resolve().parent.parent);wrapper+=['--ro-bind',root,root]
            if self.project:wrapper+=['--ro-bind',str(self.project),str(self.project)]
            wrapper+=['--proc','/proc','--dev','/dev','--tmpfs','/tmp']
            for argument in args:
                if argument.endswith('.lean'):
                    folder=str(Path(argument).resolve().parent)
                    wrapper+=['--dir',folder,'--bind',folder,folder]
            command=wrapper+['--']+command
        if not shutil.which('prlimit'):return -1,'prlimit resource isolation unavailable'
        command=['prlimit','--as='+str(4*2**30),'--cpu='+str(max(1,int(self.timeout)))+':'+str(max(2,int(self.timeout)+1)),
                 '--nproc=4096','--fsize='+str(16*2**20),'--core=0','--']+command
        process=subprocess.Popen(command,cwd=self.project,text=True,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,start_new_session=True,env=environment)
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
        if any(marker in statement for marker in ('?', '_ ')):
            return VerificationResult(False,statement,assumptions,None,diagnostics='unsupported statement metavariable')
        type_injection=any(any(separator in text for separator in ('\n','\r',':=')) for text in (statement,*assumptions))
        if type_injection or any(re.search(forbidden,text) for text in (statement,proof,*assumptions)):
            result=VerificationResult(False,statement,assumptions,None,diagnostics='forbidden declaration or trust bypass')
        elif not self.available:
            result=VerificationResult(False,statement,assumptions,None,diagnostics='Lean executable unavailable; verification not performed')
        else:
            source=''.join(f'import {module}\n' for module in self.imports)
            source+=f'set_option autoImplicit false in\ntheorem morph_candidate {" ".join(assumptions)} : {statement} := {proof}\n#print axioms morph_candidate\n'
            with tempfile.TemporaryDirectory(prefix='morph-lean-') as folder:
                path=Path(folder)/'Candidate.lean';path.write_text(source)
                code,output=self._run(['-j1',str(path)])
                _,version=self._run(['--version'])
            # Closed theorem must compile and its axiom audit must be present.
            output=re.sub(r'/tmp/morph-lean-[^/\s]+/Candidate\.lean','Candidate.lean',output)
            success=code==0 and accepted_axiom_audit(output)
            result=VerificationResult(success,statement,assumptions,hashlib.sha256(source.encode()).hexdigest() if success else None,
                                      version=version.strip(),diagnostics=output[-8000:],isolated=self.isolated,project_revision=self.project_revision)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as handle:
                handle.write(json.dumps(asdict(result))+'\n')
        return result
