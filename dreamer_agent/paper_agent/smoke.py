"""CPU synthetic joint-training mechanism experiment; never real task evidence."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import torch
from .model import Config,AgentTokenModel
from .objectives import joint_loss
from ..provenance import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(21)
    c=Config(latent_dim=4,image_tokens=2,width=32,heads=4,layers=2,targets=8)
    model=AgentTokenModel(c);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    clean=torch.randn(2,8,2,4);signal=torch.full((2,8),.7);stride=torch.full((2,8),.25)
    noisy=signal[...,None,None]*clean+(1-signal[...,None,None])*torch.randn_like(clean)
    actions=torch.stack([torch.randint(n,(2,8)) for n in c.action_sizes],-1)
    tasks=torch.tensor([[0]*8,[1]*8]);starts=torch.zeros(2,8,dtype=torch.bool);starts[:,0]=True
    valid=torch.ones_like(starts);relevant=torch.tensor([[True]*8,[False]*8]);reward=torch.zeros(2,8);reward[:,5]=1
    rows=[]
    for i in range(41):
        out=model(noisy,actions,tasks,starts,signal,stride)
        losses=joint_loss(model,out,clean,actions,reward,tasks,starts,valid,relevant,signal)
        if not all(torch.isfinite(v) for v in losses.values()):raise RuntimeError('Nonfinite loss')
        rows.append(dict(step=i,**{k:float(v.detach()) for k,v in losses.items()}))
        if i<40:
            opt.zero_grad();losses['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    model.eval()
    before=model(noisy,actions,tasks,starts,signal,stride)
    after=model(noisy,actions,(tasks+1)%c.tasks,starts,signal,stride)
    invariance=float((before['video']-after['video']).abs().max().detach())
    if invariance!=0:raise RuntimeError('Task contaminated world predictions')
    report=dict(kind='synthetic_mechanism_smoke',synthetic=True,real_success_measured=False,pretrained_weights_loaded=False,
        optimizer_updates=40,config=asdict(c),initial=rows[0],final=rows[-1],task_world_max_difference=invariance,
        code_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        limitations=['dense reference attention; no axial transformer or checkpoint conversion','fixed toy minibatch, not generalization','basic flow x-prediction; no shortcut bootstrap','no world rollout, no Minecraft dataset training'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    torch.save(dict(model=model.state_dict(),config=asdict(c),synthetic=True),a.output/'synthetic_only.pt')
    print(json.dumps(report))
if __name__=='__main__':main()
