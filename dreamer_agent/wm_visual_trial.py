"""Bounded matched branch-W pilot; no policy RL and no real-evaluation data."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from . import wm_finetune as wm
from .wm_visual import enable_decoder_grad, decode_frame, sample_last_grad, visual_losses, load_perceptual
from .provenance import sha256
from .v7_data import CACHE, MILESTONE_TASK, session_group

SKILLS = ('diamond', 'iron_ore', 'crafting_table', 'general')
VISUAL_FRAMES = (16, 20, 24, 28)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2));tmp.replace(path)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    source = CACHE / 'selection.jsonl'
    entries = [json.loads(s) for s in source.read_text().splitlines()]
    groups = {}
    for e in entries:
        groups.setdefault(e['session'].rsplit('-', 1)[-1], set()).add(session_group(e['session']))
    entries.sort(key=lambda e: e['stem'])
    _, _, spec, codec = wm._codec_only()
    records = [];used = set()
    for split, group, count, length in [('train', 'fit', 2, 32), ('eval', 'select', 1, 48)]:
        for skill in SKILLS:
            found = 0
            for e in entries:
                canonical = e['session'].rsplit('-', 1)[-1]
                if canonical in used or groups[canonical] != {group}:continue
                events = [int(x['frame']) for x in e.get('events', []) if MILESTONE_TASK.get(x['task']) == skill]
                if skill == 'general':events = [min(1000, e['frames'] // 2)]
                frames = [f for f in events if f >= 24 and f - 24 + length <= e['frames']]
                if not frames:continue
                start = frames[0] - 24
                actions_path, video = Path(e['actions']), Path(e['video'])
                ah = sha256(actions_path)
                if ah != e['actions_sha256']:raise ValueError('Source actions changed')
                vh = sha256(video)
                rgb = wm._read_frames(video, start, length)
                rows = wm._read_rows(actions_path, start, length)
                if rgb is None or rows is None:raise ValueError('Selected source failed decoding')
                actions = wm._tokens(codec, spec, rows)
                if actions is None:raise ValueError('Selected source actions unsupported')
                if sha256(video) != vh or sha256(actions_path) != ah:raise ValueError('Source changed during extraction')
                dest = output / f'{split}_{skill}_{found}.npz'
                np.savez_compressed(dest, frames=rgb, actions=actions, event_at=-1 if skill=='general' else 24)
                records.append(dict(path=str(dest.resolve()), sha256=sha256(dest), split=split, skill=skill,
                    session=e['session'],canonical_session=canonical, source_group=group,start=start,length=length,
                    video=str(video),video_sha256=vh,actions=str(actions_path),actions_sha256=ah))
                used.add(canonical);found += 1
                print('SOURCE',split,skill,e['session'],flush=True)
                if found == count:break
            if found != count:raise ValueError(f'Insufficient distinct sources for {split}/{skill}')
    manifest = dict(selection=str(source),selection_sha256=sha256(source),records=records,
        note='Sources from prior train-only V7 selection; select group is a development holdout, not final test. New hashes bind this extraction; no claim of corpus-wide alignment certification.')
    write_json(output/'manifest.json',manifest)


def load_clip(record, device):
    if sha256(record['path']) != record['sha256']:raise ValueError('Trial clip changed')
    with np.load(record['path']) as d:
        return d['frames'].copy(),torch.tensor(d['actions'].astype(np.int64),device=device),int(d['event_at'])


def prefill(tokenizer, dynamics, z, actions, generator):
    dynamics.clean_kvcache();tokenizer.decoder.clean_kvcache()
    with torch.no_grad():
        for t in range(16):
            wm._denoise(dynamics,z[:,t:t+1],actions[t:t+1],t,wm.CONTEXT_T,generator,wm.CONTEXT_STRIDE)
            decode_frame(tokenizer,z[:,t:t+1],t)


def visual_backward(tokenizer,dynamics,perceptual,z,actions,frames,seed,coefficient):
    generator=torch.Generator(device=z.device).manual_seed(seed)
    prefill(tokenizer,dynamics,z,actions,generator)
    totals=[]
    for t in range(16,32):
        active=t in VISUAL_FRAMES and coefficient>0
        if active:
            nxt=sample_last_grad(dynamics,z[:,t:t+1],actions[t-1:t],t,generator)
        else:
            nxt=wm._sample_frame(dynamics,z[:,t:t+1],actions[t-1:t],t,generator)
        with torch.set_grad_enabled(active):
            pred=decode_frame(tokenizer,nxt,t)
            if t in VISUAL_FRAMES:
                real=torch.from_numpy(frames[t]).permute(2,0,1)[None].to(z.device).float()/127.5-1
                l1,lp=visual_losses(pred,real,perceptual)
                totals.append([float(l1.detach()),float(lp.detach())])
                if active:(coefficient*(l1+lp)/len(VISUAL_FRAMES)).backward()
        with torch.no_grad():
            wm._denoise(dynamics,nxt.detach(),actions[t:t+1],t,wm.CONTEXT_T,generator,wm.CONTEXT_STRIDE)
    return np.mean(totals,axis=0).tolist()


@torch.no_grad()
def evaluate(tokenizer,dynamics,perceptual,records,output,arm):
    import cv2
    rows=[]
    for record in records:
        frames,actions,_=load_clip(record,'cuda:0')
        z=wm._encode(tokenizer,frames[:16],'cuda:0')
        for seed in (0,1):
            generator=torch.Generator(device='cuda:0').manual_seed(seed)
            prefill(tokenizer,dynamics,z,actions,generator)
            video_path=output/f'{arm}_{record["skill"]}_{seed}.mp4'
            video=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),20.,(1280,360))
            if not video.isOpened():raise RuntimeError('Video writer failed')
            metrics=[]
            try:
                for t in range(16,len(frames)):
                    nxt=wm._sample_frame(dynamics,z[:,-1:],actions[t-1:t],t,generator)
                    pred=decode_frame(tokenizer,nxt,t)
                    real=torch.from_numpy(frames[t]).permute(2,0,1)[None].cuda().float()/127.5-1
                    l1,lp=visual_losses(pred,real,perceptual)
                    err=(pred-real).abs()
                    metrics.append(dict(frame=t,l1=float(l1),lpips=float(lp),
                        center_l1=float(err[:,:,90:270,160:480].mean()),hotbar_l1=float(err[:,:,300:].mean())))
                    got=((pred[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
                    video.write(cv2.cvtColor(np.concatenate([frames[t],got],axis=1),cv2.COLOR_RGB2BGR))
                    wm._denoise(dynamics,nxt,actions[t:t+1],t,wm.CONTEXT_T,generator,wm.CONTEXT_STRIDE)
            finally:video.release()
            row=dict(arm=arm,skill=record['skill'],session=record['session'],seed=seed,
                video=str(video_path),metrics=metrics)
            rows.append(row)
            write_json(output/f'{arm}_evaluation.json',rows)
            print('EVAL',arm,record['skill'],seed,np.mean([x['l1'] for x in metrics]),flush=True)
    return rows


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(928)
    torch.cuda.set_per_process_memory_fraction(.60)
    gate=json.loads(args.gate.read_text())
    if not gate.get('passed') or gate.get('optimizer_steps')!=0:raise ValueError('Real gradient gate required')
    if gate.get('parity_frames',0)<=gate.get('decoder_window',0):raise ValueError('Cache wrap parity required')
    for name in ('wm_finetune.py','wm_visual.py'):
        if gate.get('code_hashes',{}).get(name)!=sha256(Path(__file__).parent/name):
            raise ValueError('Gradient gate code changed')
    manifest=json.loads(args.manifest.read_text());records=manifest['records']
    for r in records:
        if sha256(r['path'])!=r['sha256']:raise ValueError('Clip changed')
    train=[r for r in records if r['split']=='train'];evaluation=[r for r in records if r['split']=='eval']
    if {r['canonical_session'] for r in train}&{r['canonical_session'] for r in evaluation}:raise ValueError('Session leakage')
    order=list(range(len(train)));random.Random(928).shuffle(order)
    schedule=[order[i%len(order)] for i in range(args.updates)]
    import lpips
    perceptual_assets=[Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',
        Path(lpips.__file__).parent/'weights/v0.1/alex.pth']
    protocol=dict(scope='branch_W_engineering_pilot',arms=['base','shortcut','visual'],seed=928,
        updates_per_trained_arm=args.updates,lr=5e-5,last_n=4,lora_rank=16,alpha=32,
        visual_coefficient=.1,visual_frames=VISUAL_FRAMES,lpips_size=[180,320],context=16,
        evaluation_horizon=32,evaluation_seeds=[0,1],schedule=schedule,
        manifest_sha256=sha256(args.manifest),gate_sha256=sha256(args.gate),
        gradient_scope=gate['gradient_scope'],
        code_hashes={n:sha256(Path(__file__).parent/n) for n in ['wm_visual_trial.py','wm_visual.py','wm_finetune.py']},
        perceptual_hashes={str(p):sha256(p) for p in perceptual_assets},
        limitations=['One training seed and four development sessions; no statistical efficacy conclusion.',
            'Matched examples/updates and sampling forwards; visual backward costs extra, wall time is measured.',
            'No learned task reward or color reward; center/hotbar metrics are spatial diagnostics, not semantic truth.'])
    write_json(args.output/'protocol.json',protocol)
    tokenizer,dynamics,_,_=wm.load_backends('cuda:0')
    wm.install_lora(dynamics,4,16,32)
    initial={k:v.clone() for k,v in wm.lora_state(dynamics).items()}
    shape=(1,1,3,384//tokenizer.patch_size[0],640//tokenizer.patch_size[1])
    tokenizer.decoder.init_rope_cache(shape);enable_decoder_grad(tokenizer)
    perceptual=load_perceptual('cuda:0')
    all_results=evaluate(tokenizer,dynamics,perceptual,evaluation,args.output,'base')
    for arm in ('shortcut','visual'):
        wm.load_lora_state(dynamics,initial)
        params=[p for p in dynamics.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=0)
        started=time.monotonic()
        with (args.output/f'{arm}_metrics.jsonl').open('x') as log:
            for update,index in enumerate(schedule,1):
                frames,actions,event=load_clip(train[index],'cuda:0')
                z=wm._encode(tokenizer,frames,'cuda:0')
                generator=torch.Generator(device='cuda:0').manual_seed(928000+update)
                opt.zero_grad(set_to_none=True)
                loss=wm.clip_loss_v2(dynamics,z,actions,event,True,generator)
                l1,lp=visual_backward(tokenizer,dynamics,perceptual,z,actions,frames,1928000+update,.1 if arm=='visual' else 0.)
                if not np.isfinite([loss,l1,lp]).all() or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError('Nonfinite trial update')
                if any(p.grad is not None for p in tokenizer.parameters()) or any(p.grad is not None for p in dynamics.parameters() if not p.requires_grad):
                    raise RuntimeError('Frozen gradient leak')
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                opt.step()
                if any(not torch.isfinite(p).all() for p in params):raise FloatingPointError('Nonfinite parameters')
                row=dict(arm=arm,update=update,clip=train[index]['path'],shortcut_loss=loss,l1=l1,lpips=lp,
                    grad_norm=float(norm),elapsed_seconds=time.monotonic()-started)
                log.write(json.dumps(row)+'\n');log.flush();print('UPDATE',json.dumps(row),flush=True)
                write_json(args.output/'progress.json',row)
        opt.zero_grad(set_to_none=True)
        tmp=args.output/f'{arm}.tmp'
        torch.save(dict(lora=wm.lora_state(dynamics),opt=opt.state_dict(),update=args.updates,
            protocol=protocol,arm=arm),tmp);tmp.replace(args.output/f'{arm}.pt')
        all_results+=evaluate(tokenizer,dynamics,perceptual,evaluation,args.output,arm)
    summary=dict(status='complete',protocol_sha256=sha256(args.output/'protocol.json'),arms={},
        real_task_success_measured=False,auto_promote=False)
    for arm in protocol['arms']:
        summary['arms'][arm]={}
        for skill in SKILLS:
            rr=[r for r in all_results if r['arm']==arm and r['skill']==skill]
            summary['arms'][arm][skill]={k:float(np.mean([m[k] for r in rr for m in r['metrics']]))
                for k in ('l1','lpips','center_l1','hotbar_l1')}
    write_json(args.output/'summary.json',summary);write_json(args.output/'progress.json',summary)
    print('SUMMARY',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,required=True)
    trial=sub.add_parser('run');trial.add_argument('--output',type=Path,required=True)
    trial.add_argument('--manifest',type=Path,required=True);trial.add_argument('--gate',type=Path,required=True)
    trial.add_argument('--updates',type=int,default=16)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:
        if not 1<=a.updates<=16:raise ValueError('Pilot budget is at most 16 updates per arm')
        run(a)


if __name__=='__main__':main()
