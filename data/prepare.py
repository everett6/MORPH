"""Compile exported structured proof trajectories into versioned mapped shards.

Accepts local JSONL with family/split/text/graph/action/value. Optional real
Lean verification logs outcomes; no parser-generated or synthetic proof trust.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from morph.graph import ProofGraph,apply_action
from verifier import LeanClient
from .shards import compile_shard


def records(path,verifier=None):
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():continue
            record=json.loads(line)
            graph=ProofGraph.restore(record['graph'])
            apply_action(graph,record['action'])
            if verifier is not None and record.get('candidate_proof'):
                goal=graph.nodes[record['action']['target']]
                if not goal.formal_statement:raise ValueError('formal statement required for verification')
                result=verifier.verify(goal.formal_statement,record['candidate_proof'],goal.assumptions)
                record['verification']=asdict(result)
            yield record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--lean-project')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    verifier=LeanClient(project=args.lean_project,log_path=Path(args.output).with_suffix('.verification.jsonl')) if args.verify else None
    print(compile_shard(records(args.input,verifier),args.output))


if __name__=='__main__':main()
