"""Explicit dynamic FP16 scaling and opt-in block-scaled FP8 transport."""
import math
import torch


class DynamicLossScale:
    def __init__(self,scale=128.,growth_interval=2000):
        self.scale=float(scale);self.good_steps=0;self.growth_interval=growth_interval
    def overflow(self):
        if self.scale<=1:raise FloatingPointError('FP16 overflow at minimum loss scale; state unchanged')
        self.scale=max(1.,self.scale/2);self.good_steps=0
    def success(self):
        self.good_steps+=1
        if self.good_steps>=self.growth_interval:self.scale=min(self.scale*2,65536.);self.good_steps=0
    def state_dict(self):return dict(scale=self.scale,good_steps=self.good_steps,growth_interval=self.growth_interval)
    def load_state_dict(self,state):
        self.scale=state['scale'];self.good_steps=state['good_steps'];self.growth_interval=state['growth_interval']


def fp8_eligible(name):
    return any(name.endswith(suffix) for suffix in ('ffn.gate.weight','ffn.up.weight','ffn.down.weight'))


def pack_fp8(value,block_size=256):
    """E4M3FN bytes + FP32 per-block scale; padding is not transmitted as weights."""
    flat=value.detach().float().flatten();padding=(-flat.numel())%block_size
    padded=torch.nn.functional.pad(flat,(0,padding)).reshape(-1,block_size)
    scales=(padded.abs().amax(1)/448.).clamp_min(1e-30)
    encoded=(padded/scales[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).flatten()[:flat.numel()]
    return encoded,scales


def unpack_fp8(encoded,scales,dtype=torch.float16,block_size=256):
    size=encoded.numel();padding=(-size)%block_size
    padded=torch.nn.functional.pad(encoded,(0,padding))
    values=padded.view(torch.float8_e4m3fn).float().reshape(-1,block_size)
    return (values*scales[:,None]).to(dtype).flatten()[:size]


def pack_lm_microbatches(batches,factor):
    """Equal-sized raw LM groups preserve the mathematical effective batch.

    Uneven shapes/groups (including structured proof batches) remain unchanged;
    uniform per-micro loss averaging would otherwise weight samples incorrectly.
    """
    if factor<1:raise ValueError('packing factor must be positive')
    if factor==1 or len(batches)%factor or not batches:return batches
    if not all(isinstance(batch,torch.Tensor) and batch.ndim==2 and batch.shape==batches[0].shape for batch in batches):return batches
    return [torch.cat(batches[start:start+factor],dim=0) for start in range(0,len(batches),factor)]
