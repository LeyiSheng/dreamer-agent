"""Deterministic source selection and explicit general/task BC objective.

20% means unique training sources, not 20% per independent learner. No test
content is read. General BC carries no inferred retrospective task goal.
"""
import hashlib, math, random
from collections import Counter, defaultdict
import numpy as np

GENERAL = 'general_behavior'

def digest(text):return hashlib.sha256(text.encode()).hexdigest()

def select_sources(records, fraction, seed, tasks):
    if not 0 < fraction <= 1:raise ValueError('fraction must be in (0,1]')
    rows=[r for r in records if r['split']=='train' and r['status']=='eligible_pending_full_decode']
    budget=max(1,round(len(rows)*fraction))
    def key(r):return digest(f'{seed}:{r["stem"]}')
    buckets=defaultdict(list)
    for r in rows:buckets[(r['version'],r['identity'])].append(r)
    for bucket in buckets.values():bucket.sort(key=key)
    ordered=[]
    # Interleave aliases/versions to avoid a dominant source identity taking the budget.
    keys=sorted(buckets,key=lambda k:digest(f'{seed}:{k}'))
    cursor=0
    while True:
        current=[buckets[k][cursor] for k in keys if cursor<len(buckets[k])]
        if not current:break
        ordered.extend(current);cursor+=1
    support={t:[r for r in ordered if any(c['task']==t and not c.get('background',False) for c in r['clips'])] for t in tasks}
    selected={}
    # Guaranteed inclusion pressure for rare prerequisites, capped to leave diversity room.
    for t in sorted(tasks,key=lambda t:len(support[t])):
        quota=min(512,math.ceil(len(support[t])*fraction))
        for r in support[t][:quota]:
            if len(selected)<budget:selected[r['stem']]=r
    for r in ordered:
        if len(selected)>=budget:break
        selected.setdefault(r['stem'],r)
    return sorted(selected.values(),key=key)

def validate_splits(records):
    groups={};hashes={}
    for r in records:
        group=r['source_session'].rsplit('-',1)[-1]
        if groups.setdefault(group,r['split'])!=r['split']:raise ValueError('Session crosses splits')
        if r.get('status')=='eligible_pending_full_decode':
            h=r.get('actions_sha256')
            if not h:raise ValueError('Missing action hash')
            if hashes.setdefault(h,r['split'])!=r['split']:raise ValueError('Actions cross splits')

def objective_clips(r,tasks):
    clips=[]
    for c in r['clips']:
        if c.get('background',False):continue
        if c['task'] not in tasks:raise ValueError('Unknown task')
        if not 0<=c['start']<c['stop']<=r['frames']:raise ValueError('Invalid clip')
        clips.append(dict(c,task_id=tasks.index(c['task']),stream='task'))
    # Full source timeline provides navigation, tool transitions and pre-event context.
    # No goal inferred for these windows; general is an explicit separate output mode.
    for start in range(0,r['frames'],512):
        if r['frames']-start>=1:
            clips.append(dict(start=start,stop=min(start+512,r['frames']),task_id=len(tasks),stream='general'))
    return clips

def coverage(records,tasks):
    clips=Counter();frames=Counter();sessions=defaultdict(set);identities=defaultdict(set)
    for r in records:
        for c in objective_clips(r,tasks):
            task=tasks[c['task_id']] if c['task_id']<len(tasks) else GENERAL
            clips[task]+=1;frames[task]+=c['stop']-c['start'];sessions[task].add(r['source_session']);identities[task].add(r['identity'])
    return {t:dict(clips=clips[t],frames=frames[t],sessions=len(sessions[t]),aliases=len(identities[t])) for t in [*tasks,GENERAL]}

def window_stream(features,actions,clips,*,context,targets,sizes,seed):
    # Import the frozen window implementation, not the replaced CLI entry point.
    from .task_mix_windows import windows
    order=list(clips);random.Random(seed).shuffle(order)
    for c in order:
        for w in windows(features[c['start']:c['stop']],actions[c['start']:c['stop']],
                         context=context,targets=targets,action_sizes=sizes,task_id=c['task_id']):
            w['task_id']=c['task_id'];w['stream']=c['stream'];yield w

def shuffled_batches(iterator,batch_size,buffer_size,seed):
    if batch_size<1 or buffer_size<batch_size:raise ValueError('Invalid shuffle buffer')
    rng=random.Random(seed);buffer=[]
    for row in iterator:
        buffer.append(row)
        if len(buffer)>=buffer_size:
            rng.shuffle(buffer);batch=buffer[:batch_size];buffer=buffer[batch_size:];yield batch
    rng.shuffle(buffer)
    for start in range(0,len(buffer),batch_size):yield buffer[start:start+batch_size]


def diamond_evidence(records):
    """Counter evidence only: never convert mined/pickup counters into rollout success."""
    keys=Counter();sources=defaultdict(set)
    stages=['mine_log','craft_wooden_pickaxe','mine_cobblestone','craft_stone_pickaxe',
            'mine_iron_ore','craft_iron_ingot','craft_iron_pickaxe','mine_diamond']
    prefix=Counter()
    for r in records:
        cursor=0
        for e in sorted(r.get('events',[]),key=lambda e:e['frame']):
            if not e.get('accepted'):continue
            if e['task']=='mine_diamond':
                keys[e['key']]+=1;sources[e['key']].add(r['stem'])
            if cursor<len(stages) and e['task']==stages[cursor]:cursor+=1
        prefix[cursor]+=1
    return dict(event_keys=dict(keys),sources_per_event_key={k:len(v) for k,v in sources.items()},
        coarse_ordered_stages=stages,single_source_ordered_prefix_histogram=dict(prefix),
        interpretation='Accepted counters; pickup is separate from mining. Prefix absence may reflect recording boundaries. No cross-file stitching, no proof of complete recipes or real rollout success.')
