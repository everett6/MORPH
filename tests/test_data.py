import pytest
import torch
from data import ByteTokenizer, MappedShard, compile_shard


def test_shards_fixed_tokenizer_and_family_isolation(tmp_path):
    text='For all n, n + 0 = n. π'
    assert ByteTokenizer().decode(ByteTokenizer().encode(text))==text
    root=compile_shard([dict(family='add',split='train',text=text),dict(family='mul',split='validation',text='zero times n is zero')],tmp_path/'shard')
    shard=MappedShard(root)
    assert len(shard.records)==1 and shard.records[0]['family']=='add'
    assert shard.sample(torch.Generator().manual_seed(1),2,5).shape==(2,5)
    with pytest.raises(ValueError,match='leaks'):
        compile_shard([dict(family='same',split='train',text='a'),dict(family='same',split='validation',text='b')],tmp_path/'bad')


def test_shard_tamper_detection_and_mapped_training(tmp_path):
    import numpy as np
    from morph import MorphConfig
    from morph.training.runner import train_run
    root=compile_shard(iter([dict(family='f',split='train',text='valid theorem text')]),tmp_path/'mapped')
    shard=MappedShard(root)
    cfg=MorphConfig(vocab_size=256,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4)
    assert len(train_run(cfg,shard,1,sequence_length=4)['losses'])==1
    array=np.load(root/'0.npy',mmap_mode='r+');array[0]=100;array.flush();del array
    with pytest.raises(ValueError,match='hash'):
        MappedShard(root)
