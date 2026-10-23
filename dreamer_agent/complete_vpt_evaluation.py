"""Complete the fixed 40-episode protocol without rerunning successful episodes.

One fresh process per attempt isolates Minecraft instance-manager state. All
failed attempts are retained, never counted as policy failures or successes.
"""
import argparse,json,os,subprocess,time,traceback,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .provenance import sha256

ARMS=['baseline','rl_0','rl_1','rl_2']

def expected_jobs(protocol):
    import random
    jobs=[]
    for seed in protocol['world_seeds']:
        arms=list(ARMS);random.Random(seed).shuffle(arms)
        jobs.extend((arm,seed) for arm in arms)
    return jobs

def completed_records(records,expected):
    result={}
    for r in records:
        if r['status']!='ok':continue
        key=(r['arm'],r['world_seed'])
        if key not in expected or key in result:raise ValueError('Unexpected or duplicated successful episode')
        result[key]=r
    return result

def read_records(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

def validate_assets(protocol):
    for e in protocol['assets'].values():
        if sha256(e['path'])!=e['sha256']:raise ValueError('Protocol asset changed: '+e['path'])

def episode(protocol,arm,seed,out,port):
    out.mkdir(parents=True,exist_ok=False)
    r=None;env=None;started=time.time()
    try:
        validate_assets(protocol)
        import torch
        torch.set_num_threads(1);torch.cuda.set_device(0)
        from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
        from minerl.env.malmo import InstanceManager
        from .vpt_rl import load_vpt
        from .eval_vpt_rl_paired import evaluate_episode
        assets=protocol['assets']
        vpt,mapper,transformer=load_vpt(Path(assets['model']['path']),Path(assets['base']['path']),model_sha256=assets['model']['sha256'],weights_sha256=assets['base']['sha256'],device='cuda:0')
        if arm!='baseline':
            ck=torch.load(assets[arm]['path'],map_location='cpu',weights_only=True)
            if ck['base_weights_sha256']!=assets['base']['sha256']:raise ValueError('Wrong backbone')
            vpt.pi_head.load_state_dict(ck['pi_head'],strict=True)
        from agent import ENV_KWARGS
        InstanceManager.configure_malmo_base_port(port)
        env=HumanSurvival(**ENV_KWARGS).make()
        print('EPISODE',arm,seed,flush=True)
        r=evaluate_episode(env,vpt,mapper,transformer,seed,protocol['max_steps'],out/'steps.jsonl')
    except Exception as exc:
        r=dict(status='failed',world_seed=seed,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:
                if r is not None:r['close_error']=str(exc)
    r.update(arm=arm,attempt_directory=str(out),attempt_wall_seconds=time.time()-started)
    (out/'result.json').write_text(json.dumps(r,indent=2));print('RESULT',json.dumps(r),flush=True)
    return r['status']=='ok'

def complete(base,gpus):
    import fcntl,signal
    protocol=json.loads((base/'protocol.json').read_text());validate_assets(protocol)
    out=base/protocol.get('completion_directory','completion_20260922');out.mkdir(exist_ok=True)
    lock=(out/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected=expected_jobs(protocol);original=read_records(base/'episodes.jsonl')
    old=completed_records(original,expected)
    attempts=[]
    for p in sorted(out.glob('*/attempt_*/result.json')):attempts.append(json.loads(p.read_text()))
    done=completed_records([*original,*attempts],expected)
    pending=[j for j in expected if j not in done]
    run=dict(original_completed=len(old),remaining=len(pending),gpus=gpus,max_attempts_per_job=3,protocol_sha256=sha256(base/'protocol.json'),original_results_sha256=sha256(base/'episodes.jsonl'),runner_sha256=sha256(__file__),preserve_original_results=True)
    (out/'run.json').write_text(json.dumps(run,indent=2));print('RUN',json.dumps(run),flush=True)
    def worker(index,gpu):
        for arm,seed in pending[index::len(gpus)]:
            job=out/f'{arm}_{seed}';job.mkdir(exist_ok=True)
            existing=sorted(job.glob('attempt_*'))
            for attempt in range(len(existing)+1,4):
                target=job/f'attempt_{attempt}';env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                log=job/f'attempt_{attempt}.log'
                args=['xvfb-run','-a','-s','-screen 0 1280x720x24',os.environ.get('DREAMER_EVAL_PYTHON', sys.executable),'-m','dreamer_agent.complete_vpt_evaluation','--base',str(base),'--arm',arm,'--seed',str(seed),'--attempt-output',str(target),'--port',str(22000+index*1000+attempt*100)]
                with log.open('x') as f:
                    process=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    try:process.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                rp=target/'result.json'
                if not rp.exists():
                    target.mkdir(exist_ok=True);rp.write_text(json.dumps(dict(status='failed',arm=arm,world_seed=seed,error='Worker timed out or exited before result',returncode=process.returncode)))
                result=json.loads(rp.read_text());print('ATTEMPT',arm,seed,attempt,result['status'],flush=True)
                if result['status']=='ok':break
                time.sleep(5)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,i,g) for i,g in enumerate(gpus)]
        for f in futures:f.result()
    if sha256(base/'episodes.jsonl')!=run['original_results_sha256']:raise RuntimeError('Original results changed during completion')
    attempts=[json.loads(p.read_text()) for p in sorted(out.glob('*/attempt_*/result.json'))]
    done=completed_records([*original,*attempts],expected)
    missing=[list(k) for k in expected if k not in done]
    (out/'combined_episodes.json').write_text(json.dumps(list(done.values()),indent=2))
    summary=dict(status='complete' if not missing else 'incomplete',completed=len(done),expected=len(expected),missing=missing,failed_attempts=sum(r['status']!='ok' for r in [*original,*attempts]),arms={})
    for arm in ARMS:
        rr=[r for r in done.values() if r['arm']==arm]
        summary['arms'][arm]={t:dict(successes=sum(r['first_hit'][t] is not None for r in rr),trials=len(rr)) for t in ('log','planks','wooden_pickaxe','diamond')}
    summary['pairing_limitations']='Same world seed; initial pixel equality is not guaranteed. Infrastructure retries recorded separately.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--gpus',default='1,3');p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int);p.add_argument('--attempt-output',type=Path);p.add_argument('--port',type=int,default=22000);a=p.parse_args()
    if a.arm:episode(json.loads((a.base/'protocol.json').read_text()),a.arm,a.seed,a.attempt_output,a.port)
    else:complete(a.base,[int(g) for g in a.gpus.split(',')])
if __name__=='__main__':main()
