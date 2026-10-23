"""Reference joint objectives; no shortcut bootstrap trainer or production loader."""
import torch
from torch.nn import functional as F


def future_labels(actions,rewards,tasks,starts,valid,targets):
    b,t,k=actions.shape
    aa=torch.zeros(b,t,targets,k,dtype=actions.dtype,device=actions.device)
    rr=torch.zeros(b,t,targets,dtype=rewards.dtype,device=rewards.device)
    mm=torch.zeros(b,t,targets,dtype=torch.bool,device=actions.device)
    segment=starts.long().cumsum(1)
    for h in range(targets):
        if h>=t:break
        aa[:,:t-h,h]=actions[:,h:];rr[:,:t-h,h]=rewards[:,h:]
        # A change away and back still ends the task segment; checking endpoints is insufficient.
        okay=valid[:,:t-h].clone()
        for j in range(h+1):okay &= valid[:,j:t-h+j] & (tasks[:,j:t-h+j]==tasks[:,:t-h]) & (segment[:,j:t-h+j]==segment[:,:t-h])
        mm[:,:t-h,h]=okay
    return aa,rr,mm


def masked_mean(values,mask):
    # Preserve an explicit zero-gradient graph for an empty objective.
    return values.masked_fill(~mask,0).sum()/mask.sum().clamp_min(1)


def joint_loss(model,out,clean,actions,rewards,tasks,starts,valid,relevant,signal,reward_known=None):
    if valid.dtype!=torch.bool or relevant.dtype!=torch.bool:raise ValueError('Boolean objective masks required')
    if reward_known is None:reward_known=valid
    if reward_known.dtype!=torch.bool or reward_known.shape!=valid.shape:raise ValueError('Explicit known reward labels required')
    # Unknown values never become zero-labelled negatives: only known values are used.
    safe_reward=torch.where(reward_known,rewards,torch.zeros_like(rewards))
    aa,rr,mask=future_labels(actions,safe_reward,tasks,starts,valid,model.config.targets)
    _,_,rmask=future_labels(actions,safe_reward,tasks,starts,valid&reward_known,model.config.targets)
    policy=sum(F.cross_entropy(x.flatten(0,2),aa[...,k].flatten(),reduction='none').reshape(mask.shape) for k,x in enumerate(out['policy']))
    bc=masked_mean(policy,mask&relevant[...,None])
    reward=masked_mean(model.distribution.loss(out['reward'],rr),rmask)
    # Basic x-prediction flow loss only. Shortcut bootstrapping remains a separate integration task.
    video=(out['video']-clean).square().mean((-1,-2))*(.9*signal+.1)
    dynamics=masked_mean(video,valid&~relevant)
    return dict(total=bc+reward+dynamics,bc=bc,reward=reward,video=dynamics)
