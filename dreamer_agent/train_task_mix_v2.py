"""One synchronized task/general BC model, with resumable 20% -> full stages."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from .actions import ActionSpec
from .event_coverage import DREAMER4_TASKS
from .export_vpt import load_backend,decoded_frames,PREPROCESS
from .learning import HistoryPolicy
from .provenance import sha256
from .task_mix_data import GENERAL,select_sources,validate_splits,objective_clips,coverage,window_stream,shuffled_batches,diamond_evidence


def weighted_loss(logits,targets,valid,tasks,weights):
    losses=sum(F.cross_entropy(x.reshape(-1,x.shape[-1]),targets[...,k].reshape(-1),reduction='none').reshape(valid.shape)
               for k,x in enumerate(logits))
    mass=weights[tasks].unsqueeze(-1)*valid
    return (losses*mass).sum(),mass.sum()

def collate(rows,config,context,device):
    if not rows:
        return dict(features=torch.zeros(1,context,config['feature_dim'],device=device),
         past_actions=torch.zeros(1,context,config['past_action_dim'],device=device),
         tasks=torch.zeros(1,context,dtype=torch.long,device=device),
         targets=torch.zeros(1,context,config['targets'],len(config['action_sizes']),dtype=torch.long,device=device),
         valid=torch.zeros(1,context,config['targets'],dtype=torch.bool,device=device))
    return {k:torch.from_numpy(np.stack([r[k] for r in rows])).to(device) for k in ('features','past_actions','tasks','targets','valid')}

def compatible(metadata,current):
    for key in ('config','tasks','action_spec_sha256','extractor_sha256','context','audit_sha256'):
        if metadata.get(key)!=current.get(key):raise ValueError('Checkpoint incompatible: '+key)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--data-fraction',type=float,default=.2);ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--context',type=int,default=128)
    ap.add_argument('--targets',type=int,default=8);ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--batch-size',type=int,default=32,help='Global batch; divisible by world size')
    ap.add_argument('--buffer-size',type=int,default=64,help='Shuffle windows per rank')
    ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--weight-decay',type=float,default=.01)
    ap.add_argument('--validation-every',type=int,default=1000);ap.add_argument('--save-every',type=int,default=100)
    ap.add_argument('--validation-sources',type=int,default=64)
    group=ap.add_mutually_exclusive_group();group.add_argument('--resume',type=Path);group.add_argument('--continue-from',type=Path)
    group.add_argument('--init-bc',type=Path)
    ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args()
    rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'));local=int(os.getenv('LOCAL_RANK','0'))
    if min(a.epochs,a.context,a.targets,a.hidden,a.batch_size,a.validation_every,a.save_every,a.validation_sources)<1:raise ValueError('Positive training settings required')
    if a.batch_size%world:raise ValueError('Global batch must be divisible by GPU count')
    local_batch=a.batch_size//world
    if a.buffer_size<local_batch:raise ValueError('Shuffle buffer smaller than local batch')
    if world>1:
        from datetime import timedelta
        torch.cuda.set_device(local)
        dist.init_process_group('nccl',timeout=timedelta(hours=4))
    def barrier():
        if world>1:dist.barrier()
    def reduce(x):
        if world>1:dist.all_reduce(x)
        return x
    audit=a.audit/'training_records.jsonl';summary=json.loads((a.audit/'summary.json').read_text())
    if summary.get('status')!='audit_complete' or summary.get('tasks')!=DREAMER4_TASKS:raise ValueError('Expected audited Table4 tasks')
    records=[json.loads(l) for l in audit.open()];validate_splits(records)
    tasks=[*DREAMER4_TASKS,GENERAL]
    train=select_sources(records,a.data_fraction,a.seed,DREAMER4_TASKS)
    calrows=[r for r in records if r['split']=='calibration' and r['status']=='eligible_pending_full_decode']
    # Deterministic validation source coverage; no test data content used.
    cal=[];seen=set()
    for task in DREAMER4_TASKS:
        candidates=sorted((r for r in calrows if any(c['task']==task and not c.get('background',False) for c in r['clips'])),key=lambda r:hashlib.sha256(('val:'+r['stem']).encode()).hexdigest())
        for r in candidates[:2]:
            if r['stem'] not in seen:cal.append(r);seen.add(r['stem'])
    for r in sorted(calrows,key=lambda r:hashlib.sha256(('val:'+r['stem']).encode()).hexdigest()):
        if len(cal)>=a.validation_sources:break
        if r['stem'] not in seen:cal.append(r);seen.add(r['stem'])
    if not train or not cal:raise ValueError('Empty train/validation split')
    selected_hash=hashlib.sha256(json.dumps([r['stem'] for r in train]).encode()).hexdigest()
    cov=coverage(train,DREAMER4_TASKS)
    if any(not cov[t]['clips'] for t in DREAMER4_TASKS):raise ValueError('Selected subset lacks a task')
    from collections import Counter
    report=dict(train_sources=len(train),eligible_train_sources=sum(r['split']=='train' and r['status']=='eligible_pending_full_decode' for r in records),fraction=a.data_fraction,selection_sha256=selected_hash,
        versions=dict(Counter(r['version'] for r in train)),diamond_evidence=diamond_evidence(train),
        source_ids=[r['stem'] for r in train],validation_source_ids=[r['stem'] for r in cal],coverage=cov,
        objective='full-timeline general BC plus task-relevant BC; weighted contributions; not paper video/reward joint training')
    if rank==0:
        a.output.mkdir(parents=True,exist_ok=True)
        if not a.resume and (a.output/'config.json').exists():raise ValueError('Output already contains a run')
        if a.prepare_only:(a.output/'selection.json').write_text(json.dumps(report,indent=2))
    barrier()
    if a.prepare_only:
        if rank==0:print(json.dumps({k:v for k,v in report.items() if k not in ('source_ids','validation_source_ids')}),flush=True)
        if world>1:dist.destroy_process_group()
        return
    torch.set_num_threads(1);torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.manual_seed(a.seed)
    spec,encode,action,extractor=load_backend(Path('outputs/acceptance_20260913/assets.json'),Path('outputs/acceptance_20260913/action_probe.json'),str(device))
    config=dict(feature_dim=8192,past_action_dim=sum(spec.action_sizes),tasks=len(tasks),action_sizes=list(spec.action_sizes),hidden=a.hidden,targets=a.targets,task_conditioned=True)
    # Dataset-weighted general/task halves; capped sqrt rare-task correction.
    sizes=np.array([cov[t]['frames'] for t in DREAMER4_TASKS],dtype=float)
    tw=np.minimum(10.,np.sqrt(sizes.max()/sizes));gw=float((sizes*tw).sum()/cov[GENERAL]['frames'])
    weights=torch.tensor([*tw,gw],dtype=torch.float32,device=device)
    meta=dict(schema_version=2,config=config,tasks=tasks,context=a.context,seed=a.seed,epochs=a.epochs,
        data_fraction=a.data_fraction,world_size=world,batch_size=a.batch_size,buffer_size=a.buffer_size,selection_sha256=selected_hash,
        audit_sha256=sha256(audit),extractor_sha256=extractor,action_spec_sha256=spec.fingerprint,
        preprocessing=PREPROCESS,loss_weights=weights.tolist(),code_sha256=sha256(__file__),data_code_sha256=sha256(Path(__file__).with_name('task_mix_data.py')),windows_code_sha256=sha256(Path(__file__).with_name('task_mix_windows.py')),
        general_objective_fraction='nominal .5 by weighted frame mass; actual minibatch contributions logged',
        initialization=str(a.init_bc or a.continue_from or a.resume or 'random'),initialization_sha256=sha256(a.init_bc or a.continue_from or a.resume) if (a.init_bc or a.continue_from or a.resume) else None,validation_source_ids=report['validation_source_ids'])
    model=HistoryPolicy(**config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    start_epoch=0;skip_batches=0;updates=0;best=float('inf');restore_rng=None
    source=a.resume or a.continue_from
    if source:
        ck=torch.load(source,map_location=device,weights_only=True);compatible(ck['metadata'],meta)
        if a.resume:
            for key in ('selection_sha256','world_size','batch_size','buffer_size','seed','loss_weights','validation_source_ids','code_sha256','data_code_sha256','windows_code_sha256'):
                if ck['metadata'].get(key)!=meta.get(key):raise ValueError('Exact resume mismatch: '+key)
        if a.continue_from:
            if not ck.get('stage_complete'):raise ValueError('Expansion requires completed previous stage')
            if a.data_fraction<ck['metadata']['data_fraction']:raise ValueError('Continuation cannot shrink source fraction')
        model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer']);updates=ck['updates']
        if a.resume:start_epoch=ck['epoch'];skip_batches=ck['batch_cursor'];best=ck['best'];restore_rng=ck['rng'][rank]
    elif a.init_bc:
        ck=torch.load(a.init_bc,map_location='cpu',weights_only=True);md=ck['metadata']
        if md['action_spec_sha256']!=spec.fingerprint or md['feature_spec']['extractor_sha256']!=extractor:raise ValueError('Base BC feature/action mismatch')
        state=ck['model'];state={k:v for k,v in state.items() if k!='task.weight'}
        state['task.weight']=torch.zeros_like(model.task.weight,device='cpu')
        model.load_state_dict(state,strict=True)
    if world>1:model=DDP(model,device_ids=[local],broadcast_buffers=False)
    plain=model.module if world>1 else model
    if rank==0:
        (a.output/'config.json').write_text(json.dumps(meta,indent=2))
        (a.output/'selection.json').write_text(json.dumps(report,indent=2))
    seen_videos={}
    def source_windows(r,epoch,validation=False):
        raw=Path(r['actions_path'])
        if sha256(raw)!=r['actions_sha256']:raise ValueError('Actions changed')
        rows=[json.loads(l) for l in raw.read_text().splitlines()]
        if len(rows)!=r['frames']:raise ValueError('Action count changed')
        acts=np.stack([spec.to_local(np.asarray(action(row),dtype=np.int64)) for row in rows])
        clips=objective_clips(r,DREAMER4_TASKS)
        if validation:
            chosen={}
            for c in clips:chosen.setdefault(c['task_id'],c)
            clips=list(chosen.values())
            # Fixed, bounded validation windows; all supported tasks reported separately.
            clips=[dict(c,stop=min(c['stop'],c['start']+a.context)) for c in clips]
        needed=np.zeros(len(rows),dtype=bool)
        for c in clips:needed[c['start']:c['stop']]=True
        features=np.zeros((len(rows),8192),np.float32)
        vh=sha256(r['video_path'])
        if vh in seen_videos and seen_videos[vh]!=r['stem']:raise ValueError('Repeated video content on rank')
        seen_videos[vh]=r['stem'];count=0
        for i,frame in enumerate(decoded_frames(r['video_path'])):
            count+=1
            if i<len(rows) and needed[i]:
                features[i]=encode(frame)
                if not np.isfinite(features[i]).all():raise RuntimeError('Nonfinite frozen feature')
        if count not in (len(rows),len(rows)+1):raise ValueError('Video/action mismatch')
        if sha256(raw)!=r['actions_sha256'] or sha256(r['video_path'])!=vh:raise ValueError('Source mutated')
        yield from window_stream(features,acts,clips,context=a.context,targets=a.targets,sizes=spec.action_sizes,seed=a.seed+epoch)
    failures=0
    def stream(records,epoch,validation=False):
        nonlocal failures
        order=list(records)
        if not validation:random.Random(a.seed+epoch).shuffle(order)
        for index,r in enumerate(order):
            if index%world!=rank:continue
            try:yield from source_windows(r,epoch,validation)
            except (ValueError,OSError) as exc:
                failures+=1
                with (a.output/f'failed_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(dict(stem=r['stem'],epoch=epoch,validation=validation,error=str(exc)))+'\n')
                raise RuntimeError('Source verification failed; quarantine and re-audit explicitly: '+r['stem']) from exc
    # Validation cache is new, in-memory, independently generated for this run.
    print(json.dumps(dict(rank=rank,phase='validation_features_start',sources=len(cal))),flush=True)
    valrows=list(stream(cal,0,True))
    print(json.dumps(dict(rank=rank,phase='validation_features_ready',windows=len(valrows))),flush=True)
    def evaluate():
        plain.eval();stats=torch.zeros(len(tasks),6,device=device)
        with torch.no_grad():
            for start in range(0,len(valrows),local_batch):
                b=collate(valrows[start:start+local_batch],config,a.context,device)
                logits=plain(b['features'],b['past_actions'],b['tasks'])
                loss=sum(F.cross_entropy(x.reshape(-1,x.shape[-1]),b['targets'][...,k].reshape(-1),reduction='none').reshape(b['valid'].shape) for k,x in enumerate(logits))
                for tid in range(len(tasks)):
                    keep=b['valid']&(b['tasks']==tid).unsqueeze(-1)
                    stats[tid,0]+=loss[keep].sum();stats[tid,1]+=keep.sum()
                    valid=b['valid'][...,0]&(b['tasks']==tid);offset=0
                    for k,n in enumerate(spec.action_sizes):
                        pred=logits[k][...,0,:].argmax(-1);label=b['targets'][...,0,k]
                        prev=b['past_actions'][...,offset:offset+n];offset+=n
                        switch=valid&(prev.sum(-1)>0)&(prev.argmax(-1)!=label)
                        stats[tid,2]+=((pred==label)&valid).sum();stats[tid,3]+=valid.sum()
                        stats[tid,4]+=((pred==label)&switch).sum();stats[tid,5]+=switch.sum()
        reduce(stats);metrics={}
        for t,row in zip(tasks,stats.cpu().tolist()):
            metrics[t]=dict(nll=row[0]/row[1] if row[1] else None,labels=int(row[1]),accuracy=row[2]/row[3] if row[3] else None,switch_accuracy=row[4]/row[5] if row[5] else None)
        vals=[metrics[t]['nll'] for t in DREAMER4_TASKS if metrics[t]['nll'] is not None]
        if len(vals)!=len(DREAMER4_TASKS):raise RuntimeError('Independent validation lacks one or more tasks')
        plain.train();return float(np.mean(vals)),metrics
    def save(epoch,cursor,stage_complete=False,is_best=False):
        state=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
        rng=[None]*world
        if world>1:dist.all_gather_object(rng,state)
        else:rng=[state]
        if rank==0:
            ck=dict(model=plain.state_dict(),optimizer=optimizer.state_dict(),metadata=meta,epoch=epoch,batch_cursor=cursor,updates=updates,best=best,rng=rng,stage_complete=stage_complete)
            torch.save(ck,a.output/'latest.tmp');(a.output/'latest.tmp').replace(a.output/'latest.pt')
            if is_best:torch.save(ck,a.output/'best.tmp');(a.output/'best.tmp').replace(a.output/'best.pt')
    initial,vm=evaluate()
    if not a.resume:best=initial
    if not a.resume:save(0,0,is_best=True)
    if rank==0:
        with (a.output/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(update=updates,epoch=start_epoch,macro_nll=initial,tasks=vm,initial=True))+'\n')
    if start_epoch>=a.epochs:raise ValueError('Checkpoint already reached requested epoch budget')
    for epoch in range(start_epoch,a.epochs):
        it=iter(shuffled_batches(stream(train,epoch),local_batch,a.buffer_size,a.seed+epoch+rank))
        cursor=skip_batches if epoch==start_epoch else 0
        for _ in range(cursor):next(it,[])
        if restore_rng is not None:
            torch.set_rng_state(restore_rng['cpu'].cpu());torch.cuda.set_rng_state(restore_rng['cuda'].cpu(),device);restore_rng=None
        while True:
            rows=next(it,[]);alive=reduce(torch.tensor(int(bool(rows)),device=device))
            if not alive.item():break
            b=collate(rows,config,a.context,device);logits=model(b['features'],b['past_actions'],b['tasks'])
            numerator,mass=weighted_loss(logits,b['targets'],b['valid'],b['tasks'],weights)
            denominator=reduce(mass.detach().clone())
            if denominator.item()<=0:raise RuntimeError('Empty global batch')
            loss=numerator*(world/denominator)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(plain.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            updates+=1;cursor+=1
            total=reduce(numerator.detach().clone());actual=reduce(torch.tensor(len(rows),device=device))
            general_mass=reduce((weights[b['tasks']].unsqueeze(-1)*b['valid']*(b['tasks']==len(DREAMER4_TASKS)).unsqueeze(-1)).sum())
            if rank==0:
                row=dict(update=updates,epoch=epoch+1,batch_cursor=cursor,weighted_nll=float(total/denominator),global_windows=int(actual),general_weight_fraction=float(general_mass/denominator),data_fraction=a.data_fraction)
                with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if updates%10==0:print(json.dumps(row),flush=True)
            improved=False
            if updates%a.validation_every==0:
                metric,vm=evaluate();improved=metric<best;best=min(best,metric)
                if rank==0:
                    with (a.output/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(update=updates,epoch=epoch+1,macro_nll=metric,tasks=vm))+'\n')
            if updates%a.save_every==0 or improved:save(epoch,cursor,is_best=improved)
        metric,vm=evaluate();improved=metric<best;best=min(best,metric)
        if rank==0:
            with (a.output/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(update=updates,epoch=epoch+1,macro_nll=metric,tasks=vm,epoch_end=True))+'\n')
        save(epoch+1,0,stage_complete=epoch+1==a.epochs,is_best=improved)
    total_failures=int(reduce(torch.tensor(failures,device=device)))
    if rank==0:
        (a.output/'summary.json').write_text(json.dumps(dict(status='complete_with_exclusions' if total_failures else 'complete',updates=updates,epochs=a.epochs,source_failures=total_failures,checkpoint='latest.pt',data_fraction=a.data_fraction,real_success_measured=False),indent=2))
    if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
