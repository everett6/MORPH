"""Compile value-only supervision from measured model/verifier search outcomes."""
import hashlib
import json
from pathlib import Path
from morph.inference_payload import payload_prompt


def search_value_records(search):
    events=search['events']
    for index,event in enumerate(events):
        if 'prompt_graph' not in event:raise ValueError('search event lacks its actual observed input graph')
        graph=event['prompt_graph'];goal=graph['nodes'][0]['formal_statement']
        family=hashlib.sha256(' '.join(goal.split()).encode()).hexdigest()
        remaining=events[index:]
        compiler_cost=sum(item.get('compiler_seconds',0.) for item in remaining)
        target=float(search['success'])/(1+len(remaining)-1+compiler_cost)
        prompt=payload_prompt(goal,diagnostics=event['diagnostics_context'])
        yield dict(family=family,split='train',text=prompt+event['decoded']['raw']+'\0',graph=graph,
                   action=dict(type='VERIFY',target=0),value=target,value_only=True,trace_kind='measured_search_outcome',
                   measured=dict(success=search['success'],remaining_attempts=len(remaining),compiler_seconds=compiler_cost,
                                 generated_tokens=sum(item['decoded'].get('generated_tokens',0) for item in remaining)),
                   value_formula='terminal_success / (remaining_attempts + remaining_compiler_seconds)')


def write_search_values(searches,path):
    rows=[row for search in searches for row in search_value_records(search)]
    Path(path).write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n')
    return len(rows)
