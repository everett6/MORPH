import json
from pathlib import Path
import pytest
import torch
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from morph.training.proof_runner import run_proof_training,collate_proof,load_examples


def cfg():return DeltaConfig(d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4,d_graph=4,graph_nodes=3,
                             expert_count=3,expert_depth=1,expert_ffn=16,latent_dim=4,memory_dim=4,statement_field_length=32)


def traces(tmp_path):
    records=[json.loads(line) for line in Path('docs/proof_examples.jsonl').read_text().splitlines()]
    path=tmp_path/'traces.jsonl';path.write_text('\n'.join(json.dumps(row) for row in records)+'\n');return path


def test_delta_joint_disk_training_and_canonical_inference(tmp_path):
    torch.set_num_threads(2);path=traces(tmp_path)
    model,losses=run_proof_training(cfg(),path,2,engine='disk',state_dir=tmp_path/'run'/'state',sequence_length=128)
    assert len(losses)==2 and model.executor.store.manifest['step']==2
    batch=collate_proof(load_examples(path,'train')[:1],cfg(),128)
    output=model.forward_proof(batch['tokens'],batch['graph'])
    reference=DeltaModel(cfg())
    with torch.no_grad():
        for name,p in reference.named_parameters():p.copy_(model.executor.store.weight(name))
        expected=reference.forward_proof(batch['tokens'],batch['graph'])
    torch.testing.assert_close(output['logits'],expected['logits'],atol=1e-6,rtol=1e-5)
    assert output['states'] is None and 'action_logits' in output
    model.close()


def test_delta_proof_payload_is_never_silently_truncated(tmp_path):
    record=load_examples(traces(tmp_path),'train')[0];record['payload_start']=5
    with pytest.raises(ValueError,match='truncat|sequence'):
        collate_proof([record],cfg(),8)


def test_delta_proof_cli_meta_preflight_and_warmstart(tmp_path):
    from streamtrain.delta_proof import proof_preflight
    report=proof_preflight(cfg(),traces(tmp_path),tmp_path/'run',device='cpu',precision='fp32',sequence_length=128)
    assert report['parameters']>0 and report['canonical_state_bytes']==report['parameters']*12


def test_delta_proof_export_roundtrip(tmp_path):
    from streamtrain.export import export_fp16,load_fp16
    model,_=run_proof_training(cfg(),traces(tmp_path),1,engine='disk',state_dir=tmp_path/'run'/'state',sequence_length=128)
    model.close()
    export_fp16(tmp_path/'run'/'state',tmp_path/'export')
    loaded=load_fp16(tmp_path/'export',device='cpu')
    batch=collate_proof(load_examples(traces(tmp_path),'train')[:1],cfg(),128)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):
        output=loaded.forward_proof(batch['tokens'],batch['graph'])
    assert torch.isfinite(output['logits']).all() and output['action_logits'].shape[0]==1


def test_explicit_lm_phase_transition_retains_state_and_clocks(tmp_path):
    from streamtrain.disk import DiskState
    root=tmp_path/'run'/'state'
    with torch.device('meta'):template=DeltaModel(cfg())
    store=DiskState.create(root,template)
    store.manifest['step']=3
    from streamtrain.disk import durable_json
    durable_json(root/'manifest.json',store.manifest)
    with pytest.raises(ValueError,match='settings'):
        run_proof_training(cfg(),traces(tmp_path),4,engine='disk',state_dir=root,resume=True,sequence_length=128)
    model,losses=run_proof_training(cfg(),traces(tmp_path),4,engine='disk',state_dir=root,resume=True,sequence_length=128,continue_lm_state=True)
    assert len(losses)==1 and model.executor.store.manifest['step']==4
    assert model.executor.store.manifest['phase_transition']['step']==3
    model.close()


def test_proof_cli_backend_matches_lm_transition_configuration():
    from streamtrain.delta_proof import load_config
    assert load_config('configs/morph_delta_tiny.json').ssm_backend=='triton'
    assert load_config('configs/morph_delta_tiny.json','parallel').ssm_backend=='parallel'
