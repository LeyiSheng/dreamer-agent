"""Dense reference of causal one-way agent-token attention and MTP heads.

This is an independently implemented mechanism prototype, NOT the paper's
space/time transformer or an adapter for released Dreamerv4-MC checkpoints.
Inputs are latents, not pixels; all weights start fresh unless explicitly loaded.
"""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    latent_dim:int=32
    image_tokens:int=4
    width:int=64
    heads:int=4
    layers:int=2
    tasks:int=20
    targets:int=8
    registers:int=4
    max_time:int=256
    reward_bins:int=255
    symlog_limit:float=20.
    # Engineering defaults for distribution support; paper does not fix these here.
    action_sizes:tuple=(2,)*23+(121,)

    def __post_init__(self):
        if min(self.latent_dim,self.image_tokens,self.width,self.heads,self.layers,self.tasks,self.targets,self.max_time)<1:raise ValueError('Positive dimensions required')
        if self.width%self.heads or self.reward_bins<3 or self.symlog_limit<=0 or self.registers<0:raise ValueError('Invalid attention/distribution configuration')
        if not self.action_sizes or min(self.action_sizes)<2:raise ValueError('Invalid categorical actions')


def one_way_mask(starts,world_tokens):
    """SDPA True=allowed. Agents read causal world/agents; world never reads agents.

    Distinct episode segments have no mutual attention, even inside one batch.
    Current-step action labels are NEVER tokens: forward constructs a_<t.
    """
    if starts.ndim!=2 or starts.dtype!=torch.bool or not starts[:,0].all():raise ValueError('Explicit episode starts required')
    b,t=starts.shape;n=world_tokens+1
    time=torch.arange(t,device=starts.device).repeat_interleave(n)
    agent=torch.arange(t*n,device=starts.device)%n==world_tokens
    segment=starts.long().cumsum(1).repeat_interleave(n,1)
    allowed=(time[None,:]<=time[:,None]) & (agent[:,None] | ~agent[None,:])
    return (allowed[None]& (segment[:,:,None]==segment[:,None,:]))[:,None]


def shifted_actions(actions,starts):
    previous=torch.zeros_like(actions);previous[:,1:]=actions[:,:-1]
    known=~starts;previous=previous.masked_fill(starts[...,None],0)
    return previous,known


class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.heads=c.heads;self.norm1=nn.LayerNorm(c.width);self.norm2=nn.LayerNorm(c.width)
        self.qkv=nn.Linear(c.width,3*c.width);self.output=nn.Linear(c.width,c.width)
        self.mlp=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),nn.Linear(4*c.width,c.width))
    def forward(self,x,mask):
        b,l,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,l,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        h=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.).transpose(1,2).reshape(b,l,d)
        x=x+self.output(h);return x+self.mlp(self.norm2(x))


class TwoHot(nn.Module):
    def __init__(self,bins=255,limit=20.):
        super().__init__();self.register_buffer('support',(2*torch.arange(bins,dtype=torch.float32)-(bins-1))/(bins-1)*limit)
    def target(self,value):
        value=value.detach()
        if not torch.isfinite(value).all():raise ValueError('Nonfinite scalar target')
        bins=self.support.sign()*torch.expm1(self.support.abs())
        value=value.clamp(bins[0],bins[-1])
        high=torch.searchsorted(bins,value.contiguous(),right=True).clamp(1,len(bins)-1)
        low=high-1
        weight=(value-bins[low])/(bins[high]-bins[low])
        out=torch.zeros(*value.shape,len(self.support),device=value.device,dtype=value.dtype)
        out.scatter_add_(-1,low[...,None],(1-weight)[...,None]);out.scatter_add_(-1,high[...,None],weight[...,None]);return out
    def loss(self,logits,value):return -(self.target(value)*logits.log_softmax(-1)).sum(-1)
    def mean(self,logits):
        # Expected scalar in the original symexp-spaced support, not symexp(E[symlog]).
        scalar=self.support.sign()*torch.expm1(self.support.abs())
        p=logits.softmax(-1);half=len(scalar)//2
        # Pair symmetric bins before summation to avoid a biased zero-reward prediction.
        return ((p[...,-half:]-p[...,:half].flip(-1))*scalar[-half:]).sum(-1)


class AgentTokenModel(nn.Module):
    def __init__(self,c=Config()):
        super().__init__();self.config=c
        self.image=nn.Linear(c.latent_dim,c.width)
        self.action=nn.ModuleList(nn.Embedding(n,c.width) for n in c.action_sizes)
        self.action_known=nn.Embedding(2,c.width)
        self.noise=nn.Linear(2,c.width)
        self.register=nn.Parameter(torch.zeros(c.registers,c.width))
        self.task=nn.Embedding(c.tasks,c.width)
        self.world_tokens=c.image_tokens+len(c.action_sizes)+1+c.registers
        self.slot=nn.Parameter(torch.randn(self.world_tokens+1,c.width)*.01)
        self.time=nn.Embedding(c.max_time,c.width)
        self.blocks=nn.ModuleList(Block(c) for _ in range(c.layers));self.norm=nn.LayerNorm(c.width)
        def head(outputs):return nn.Sequential(nn.Linear(c.width,c.width),nn.SiLU(),nn.Linear(c.width,outputs))
        self.video=nn.Linear(c.width,c.latent_dim)
        self.policy=nn.ModuleList(head(c.targets*n) for n in c.action_sizes)
        self.reward=head(c.targets*c.reward_bins);self.value=head(c.reward_bins)
        self.distribution=TwoHot(c.reward_bins,c.symlog_limit)

    def forward(self,latents,actions,tasks,starts,signal,stride):
        c=self.config;b,t,n,d=latents.shape
        if (n,d)!=(c.image_tokens,c.latent_dim) or t>c.max_time:raise ValueError('Latent shape/time mismatch')
        if actions.shape!=(b,t,len(c.action_sizes)) or tasks.shape!=(b,t) or starts.shape!=(b,t):raise ValueError('Sequence shapes disagree')
        if signal.shape!=(b,t) or stride.shape!=(b,t):raise ValueError('Noise parameter shape mismatch')
        if not all(torch.isfinite(x).all() for x in (latents,signal,stride)):raise ValueError('Nonfinite inputs')
        if (signal<0).any() or (signal>1).any() or (stride<=0).any() or (stride>1).any():raise ValueError('Invalid flow parameters')
        for k,size in enumerate(c.action_sizes):
            if ((actions[...,k]<0)|(actions[...,k]>=size)).any():raise ValueError('Invalid action ID')
        previous,known=shifted_actions(actions,starts)
        act=torch.stack([emb(previous[...,k])*known[...,None] for k,emb in enumerate(self.action)],dim=2)
        act=act+self.action_known(known.long())[:,:,None]
        parts=[self.image(latents),act,self.noise(torch.stack([signal,stride],-1))[:,:,None],self.register[None,None].expand(b,t,-1,-1),self.task(tasks)[:,:,None]]
        x=torch.cat(parts,2)
        indices=torch.arange(t,device=latents.device)[None].expand(b,t)
        origins=torch.where(starts,indices,torch.zeros_like(indices)).cummax(1).values
        x=x+self.slot[None,None]+self.time(indices-origins)[:,:,None]
        x=x.reshape(b,t*(self.world_tokens+1),c.width);mask=one_way_mask(starts,self.world_tokens)
        for block in self.blocks:x=block(x,mask)
        x=self.norm(x.reshape(b,t,self.world_tokens+1,c.width));h=x[:,:,-1]
        return dict(video=self.video(x[:,:,:c.image_tokens]),policy=[head(h).reshape(b,t,c.targets,size) for head,size in zip(self.policy,c.action_sizes)],
                    reward=self.reward(h).reshape(b,t,c.targets,c.reward_bins),value=self.value(h),agent=h)

    def freeze_for_rl(self):
        self.eval().requires_grad_(False)
        self.policy.requires_grad_(True);self.value.requires_grad_(True)
        # No prior is silently created; caller must bind a separate fixed snapshot.
        return self
