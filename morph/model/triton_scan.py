"""Fused recurrent CUDA scan with chunk-checkpointed recomputation backward.

One program per batch/head/channel; reductions across channels use FP32 atomics.
This is an optional correctness/measurement prototype, not a blanket speed claim.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _forward(X,DT,A,B,C,S,Y,FINAL,CHECK,T:tl.constexpr,H:tl.constexpr,P:tl.constexpr,N:tl.constexpr,CHUNK:tl.constexpr,NS:tl.constexpr):
    program=tl.program_id(0);p=program%P;h=(program//P)%H;batch=program//(P*H)
    n=tl.arange(0,NS);valid=n<N
    state=tl.load(S+program*N+n,valid,0).to(tl.float32)
    tl.store(CHECK+program*N+n,state,valid)
    a=tl.load(A+h).to(tl.float32)
    for t in range(T):
        dt=tl.load(DT+(batch*T+t)*H+h).to(tl.float32);decay=tl.exp(dt*a);integral=libdevice.expm1(dt*a)/a
        x=tl.load(X+((batch*T+t)*H+h)*P+p).to(tl.float32)
        offset=((batch*T+t)*H+h)*N+n
        b=tl.load(B+offset,valid,0).to(tl.float32);c=tl.load(C+offset,valid,0).to(tl.float32)
        state=decay*state+integral*x*b
        tl.store(Y+((batch*T+t)*H+h)*P+p,tl.sum(state*c,0))
        if (t+1)%CHUNK==0 or t==T-1:
            checkpoint=(t//CHUNK)+1
            tl.store(CHECK+((checkpoint*tl.num_programs(0)+program)*N+n),state,valid)
    tl.store(FINAL+program*N+n,state,valid)


@triton.jit
def _backward(X,DT,A,B,C,CHECK,DY,DF,DX,DDT,DA,DB,DC,DS,T:tl.constexpr,H:tl.constexpr,P:tl.constexpr,N:tl.constexpr,CHUNK:tl.constexpr,NS:tl.constexpr):
    program=tl.program_id(0);p=program%P;h=(program//P)%H;batch=program//(P*H)
    n=tl.arange(0,NS);rows=tl.arange(0,CHUNK);valid=n<N
    carry=tl.load(DF+program*N+n,valid,0).to(tl.float32);a=tl.load(A+h).to(tl.float32)
    ga=0.
    for reverse_chunk in range(tl.cdiv(T,CHUNK)):
        chunk=tl.cdiv(T,CHUNK)-1-reverse_chunk;start=chunk*CHUNK
        initial=tl.load(CHECK+((chunk*tl.num_programs(0)+program)*N+n),valid,0).to(tl.float32)
        state=initial;history=tl.full((CHUNK,NS),0,tl.float32)
        for local in range(CHUNK):
            t=start+local
            dt=tl.load(DT+(batch*T+t)*H+h,t<T,0).to(tl.float32);decay=tl.exp(dt*a);integral=libdevice.expm1(dt*a)/a
            x=tl.load(X+((batch*T+t)*H+h)*P+p,t<T,0).to(tl.float32)
            offset=((batch*T+t)*H+h)*N+n
            b=tl.load(B+offset,valid&(t<T),0).to(tl.float32)
            state=decay*state+integral*x*b
            history=tl.where(rows[:,None]==local,state[None,:],history)
        for reverse_local in range(CHUNK):
            local=CHUNK-1-reverse_local;t=start+local
            current=tl.sum(tl.where(rows[:,None]==local,history,0.),0)
            previous=tl.sum(tl.where(rows[:,None]==local-1,history,0.),0)
            previous=tl.where(local==0,initial,previous)
            dt=tl.load(DT+(batch*T+t)*H+h,t<T,0).to(tl.float32);decay=tl.exp(dt*a);integral=libdevice.expm1(dt*a)/a
            x=tl.load(X+((batch*T+t)*H+h)*P+p,t<T,0).to(tl.float32)
            offset=((batch*T+t)*H+h)*N+n
            b=tl.load(B+offset,valid&(t<T),0).to(tl.float32);c=tl.load(C+offset,valid&(t<T),0).to(tl.float32)
            dy=tl.load(DY+((batch*T+t)*H+h)*P+p,t<T,0).to(tl.float32)
            gradient=carry+dy*c
            ge=tl.sum(gradient*previous,0);gk=tl.sum(gradient*x*b,0)
            tl.store(DX+((batch*T+t)*H+h)*P+p,tl.sum(gradient*integral*b,0),t<T)
            tl.atomic_add(DDT+(batch*T+t)*H+h,ge*a*decay+gk*decay,t<T)
            z=dt*a
            derivative=tl.where(tl.abs(z)<1e-3,dt*dt*(.5+z/3.+z*z/8.),(dt*decay*a-libdevice.expm1(z))/(a*a))
            ga+=tl.where(t<T,ge*dt*decay+gk*derivative,0.)
            tl.atomic_add(DB+offset,gradient*integral*x,valid&(t<T))
            tl.atomic_add(DC+offset,dy*current,valid&(t<T))
            carry=tl.where(t<T,gradient*decay,carry)
    tl.atomic_add(DA+h,ga)
    tl.store(DS+program*N+n,carry,valid)


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,dt,a,b,c,state,chunk):
        args=[value.contiguous() for value in (x,dt,a,b,c,state)]
        x,dt,a,b,c,state=args;batch,length,heads,channels=x.shape;n=state.shape[-1]
        y=torch.empty_like(x);final=torch.empty_like(state)
        check=torch.empty((triton.cdiv(length,chunk)+1,batch,heads,channels,n),device=x.device,dtype=torch.float32)
        programs=batch*heads*channels
        _forward[(programs,)](*args,y,final,check,length,heads,channels,n,chunk,triton.next_power_of_2(n),num_warps=4)
        ctx.save_for_backward(*args,check);ctx.chunk=chunk
        return y,final

    @staticmethod
    def backward(ctx,dy,df):
        x,dt,a,b,c,state,check=ctx.saved_tensors
        batch,length,heads,channels=x.shape;n=state.shape[-1]
        if df is None:df=torch.zeros_like(state)
        if dy is None:dy=torch.zeros_like(x)
        grads=[torch.zeros_like(value,dtype=torch.float32) for value in (x,dt,a,b,c,state)]
        _backward[(batch*heads*channels,)](x,dt,a,b,c,check,dy.contiguous(),df.contiguous(),*grads,
                    length,heads,channels,n,ctx.chunk,triton.next_power_of_2(n),num_warps=4)
        return *(gradient.to(value.dtype) for gradient,value in zip(grads,(x,dt,a,b,c,state))),None


def triton_scan(x,dt,a,b,c,state,chunk_size=16):
    if not x.is_cuda or x.dtype!=torch.float32:raise ValueError('Triton reference requires CUDA FP32 recurrence')
    return _Scan.apply(x,dt,a,b,c,state,chunk_size)
