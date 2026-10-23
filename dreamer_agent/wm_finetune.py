"""Branch W: LoRA finetune of the last Dreamerv4-MC dynamics blocks (PLAN_V7 §10).

The frozen world model drifts off rare, task-relevant frames (diamond ore, iron ore,
crafting). This trains a low-rank update so those frames stay predictable from offline
VPT video. The tokenizer stays frozen. Real evaluation trajectories are never read.

  python -m dreamer_agent.wm_finetune prepare --output /dev/shm/v7_wm_ft_20260927
  python -m dreamer_agent.wm_finetune train --clips /dev/shm/v7_wm_ft_20260927 --output outputs/v7_wm_ft_20260927
  python -m dreamer_agent.wm_finetune probe --clips /dev/shm/v7_wm_ft_20260927 --lora outputs/v7_wm_ft_20260927/lora.pt
"""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .actions import ActionSpec
from .provenance import sha256
from .v7_data import CACHE, MILESTONE_TASK, session_group

ROOT = Path(__file__).resolve().parents[1]
CLIP = 32
CONTEXT = 16          # real frames written into the KV cache before the loss frames
EVENT_INDEX = 24      # event sits here, so the clip is [event-24, event+8)
K_SAMPLES = 64
STRIDE_INDEX = 2      # log2 of the 4-step sampler used at imagination time
CONTEXT_STRIDE = 3    # adapter context/rewrite passes use log2(steps) + 1
PROBE_HORIZON = 96    # generated frames per long probe clip
PROBE_CAPS = dict(diamond=24, iron_ore=12)
CONTEXT_T = 48        # 0.75 * 64, same signal the adapter uses for real context
TRAIN_T = (0, 16, 32, 48)
RARE = ('diamond', 'iron_ore', 'iron_pickaxe', 'furnace', 'crafting_table', 'iron_ingot')
CAPS = dict(diamond=800, iron_ore=600, iron_pickaxe=400, furnace=300, crafting_table=300,
            wooden_pickaxe=250, stone_pickaxe=250, iron_ingot=300, cobblestone=200, log=200,
            planks=150, stick=150, general=400)
PROBE_CAP = 24
WEIGHT = dict(diamond=6, iron_ore=4, iron_pickaxe=4, furnace=3, crafting_table=3, iron_ingot=3,
              wooden_pickaxe=2, stone_pickaxe=2, cobblestone=1, log=1, planks=1, stick=1, general=1)


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        dtype, device = base.weight.dtype, base.weight.device
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x.reshape(-1, x.shape[-1]) @ self.A.T @ self.B.T).reshape(*x.shape[:-1], -1) * self.scale


def install_lora(dynamics, last_n, rank, alpha):
    """LoRA on spatial attention, FFN and the output head of the last blocks.

    Temporal attention uses flash_attn_with_kvcache, which has no backward. Its output is
    detached, so those layers keep the pretrained behavior and the spatial/FFN path still trains.
    """
    if not 1 <= last_n <= len(dynamics.blocks) or rank < 1 or alpha <= 0:
        raise ValueError('Invalid LoRA block/rank/alpha configuration')
    dynamics.requires_grad_(False)
    for block in dynamics.blocks[-last_n:]:
        for name in ('q', 'k', 'v', 'o'):
            setattr(block.attn, name, LoRALinear(getattr(block.attn, name), rank, alpha))
        for name in ('w1', 'w2', 'w3'):
            setattr(block.ffn, name, LoRALinear(getattr(block.ffn, name), rank, alpha))
        if block.attn_temporal is not None:
            block.attn_temporal.register_forward_hook(lambda _m, _i, out: out.detach())
    dynamics.head.head = LoRALinear(dynamics.head.head, rank, alpha)
    dynamics.blocks[-last_n].register_forward_pre_hook(
        lambda _m, args: tuple(a.detach() if torch.is_tensor(a) else a for a in args))


def lora_state(dynamics):
    return {n: p.detach().cpu() for n, p in dynamics.named_parameters() if p.requires_grad}


def load_lora_state(dynamics, state):
    own = dict(dynamics.named_parameters())
    expected = {k for k, p in own.items() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('LoRA checkpoint must contain exactly the installed trainable parameters')
    if any(state[k].shape != own[k].shape or not torch.isfinite(state[k]).all() for k in state):
        raise ValueError('LoRA checkpoint has incompatible shapes or nonfinite parameters')
    with torch.no_grad():
        for k, v in state.items():
            own[k].copy_(v.to(own[k].device, dtype=own[k].dtype))


def load_backends(device):
    assets = json.loads((ROOT / 'outputs/acceptance_20260913/assets.json').read_text())
    for entry in assets['files']:
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('WM assets changed')
    snapshot = Path(assets['files'][0]['path']).parents[1]
    sys.path.insert(0, str(ROOT / 'Dreamerv4-MC'))
    from src.modules.actokenizer import MineCraftActionTokenizer
    from src.modules.dynamic_model import DynamicModel
    from src.modules.kvcache import KVCacheFullyStatic
    from src.modules.tokenizer import CausalTokenizer

    # The inference cache stores K/V in place. Detach so a later frame cannot
    # overwrite tensors the backward of the current frame still needs, and so
    # the in-place write does not break autograd.
    def store(self, k, v, idx):
        idx = idx % self.max_len
        self.k[idx] = k[0].detach()
        self.v[idx] = v[0].detach()
    KVCacheFullyStatic.store_kvcache = store

    models = []
    for cls, sub in ((CausalTokenizer, 'tokenizer'), (DynamicModel, 'dynamic')):
        net, info = cls.from_pretrained(str(snapshot), subfolder=sub, local_files_only=True,
                                        torch_dtype=torch.bfloat16, output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError(f'Strict WM loading failed for {sub}')
        models.append(net.to(device).eval())
    tokenizer, dynamics = models
    tokenizer.requires_grad_(False)
    dynamics.requires_grad_(False)
    spec = ActionSpec.from_dict(json.loads((ROOT / 'outputs/acceptance_20260913/action_probe.json').read_text())['action_spec'])
    dynamics.init_rope_cache(frame_seqlen=tokenizer.encoder.num_learnable_tokens // 2, action_slot_len=len(spec.slots))
    return tokenizer, dynamics, spec, MineCraftActionTokenizer()


def _encode(tokenizer, frames, device):
    pixels = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    pixels = F.pad(pixels, (0, 0, 12, 12))[None].to(device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        encoded = tokenizer.encoder(pixels)
    z = encoded.reshape(1, frames.shape[0], encoded.shape[2] // 2, encoded.shape[3] * 2)
    return z


def _denoise(dynamics, latent, action, frame_idx, t_index, generator, stride=STRIDE_INDEX):
    t = t_index / K_SAMPLES
    noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)
    noisy = latent * t + noise * (1 - t)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = dynamics(img=noisy, timestep=torch.tensor([t_index], device=latent.device),
                        timestep_stride=torch.tensor([stride], device=latent.device),
                        action_ids=action, current_frame_idx=torch.tensor([frame_idx], device=latent.device),
                        read_rope_indices=dynamics.get_read_rope_indices(frame_idx))
    return pred


FLOW_STRIDE = 6                 # d = 1/64: the only step size trained against the real latent
SHORTCUT_STRIDES = (2, 3)       # 4-step sampler and the context/rewrite stride
SHORTCUT_P = (0.7, 0.3)
P_SHORTCUT = 0.5


def _pred(dynamics, x, action, frame_idx, t_index, stride):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return dynamics(img=x, timestep=torch.tensor([t_index], device=x.device),
                        timestep_stride=torch.tensor([stride], device=x.device),
                        action_ids=action, current_frame_idx=torch.tensor([frame_idx], device=x.device),
                        read_rope_indices=dynamics.get_read_rope_indices(frame_idx))


@torch.no_grad()
def _sample_frame(dynamics, like, action, frame_idx, generator):
    """The imagination-time 4-step sampler (generate_one_frame, steps=4)."""
    noise = torch.randn(like.shape, device=like.device, dtype=like.dtype, generator=generator)
    x, t = noise, 0.0
    for _ in range(4):
        x_hat = _pred(dynamics, x, action, frame_idx, int(t * K_SAMPLES), STRIDE_INDEX)
        t += 0.25
        x = x_hat * t + noise * (1 - t)
    return x


def _frame_objective(dynamics, z, action, frame_idx, generator):
    """Dreamer 4 shortcut objective for one frame (x-prediction):
    flow: stride 6, tau uniform on the 1/64 grid, target = real latent, ramp weight 0.9*tau + 0.1;
    shortcut: stride k, target = two stop-gradient half steps at stride k+1, weighted (1-tau)^2 in
    flow space (= plain MSE in x space). Large steps are never regressed onto the real latent,
    which would average over futures and blur rare detail."""
    u = float(torch.rand((), device=z.device, generator=generator))
    noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
    if u < P_SHORTCUT:
        k = SHORTCUT_STRIDES[0] if float(torch.rand((), device=z.device, generator=generator)) < SHORTCUT_P[0] else SHORTCUT_STRIDES[1]
        grid = K_SAMPLES >> k
        t_index = int(torch.randint(K_SAMPLES // grid, (), device=z.device, generator=generator)) * grid
        tau = t_index / K_SAMPLES
        x_tau = z * tau + noise * (1 - tau)
        with torch.no_grad():
            half = grid // 2
            x1 = _pred(dynamics, x_tau, action, frame_idx, t_index, k + 1).float()
            b1 = (x1 - x_tau.float()) / (1 - tau)
            x_mid = x_tau.float() + b1 * (half / K_SAMPLES)
            tau2 = (t_index + half) / K_SAMPLES
            x2 = _pred(dynamics, x_mid.to(z.dtype), action, frame_idx, t_index + half, k + 1).float()
            b2 = (x2 - x_mid) / (1 - tau2)
            target = x_tau.float() + (b1 + b2) / 2 * (1 - tau)
        pred = _pred(dynamics, x_tau, action, frame_idx, t_index, k)
        return F.mse_loss(pred.float(), target)
    t_index = int(torch.randint(K_SAMPLES, (), device=z.device, generator=generator))
    tau = t_index / K_SAMPLES
    pred = _pred(dynamics, z * tau + noise * (1 - tau), action, frame_idx, t_index, FLOW_STRIDE)
    return (0.9 * tau + 0.1) * F.mse_loss(pred.float(), z.float())


def clip_loss_v2(dynamics, z, actions, event_at, rollout, generator):
    """Per-frame shortcut objective on frames CONTEXT..CLIP-1; the block of frame t carries a_{t-1}
    while it is generated and is rewritten with a_t afterwards, as in AlignedWorldAdapter. With
    `rollout`, the frame written back is the model's own 4-step sample, so later frames are trained
    from drifted context toward the real latent."""
    dynamics.clean_kvcache()
    with torch.no_grad():
        for i in range(CONTEXT):
            _denoise(dynamics, z[:, i:i + 1], actions[i:i + 1], i, CONTEXT_T, generator, CONTEXT_STRIDE)
    weights = [3.0 if event_at >= 0 and abs(t - event_at) <= 4 else 1.0 for t in range(CONTEXT, CLIP)]
    norm = sum(weights)
    value = 0.0
    for j, t in enumerate(range(CONTEXT, CLIP)):
        loss = weights[j] / norm * _frame_objective(dynamics, z[:, t:t + 1], actions[t - 1:t], t, generator)
        loss.backward()
        value += float(loss.detach())
        nxt = _sample_frame(dynamics, z[:, t:t + 1], actions[t - 1:t], t, generator) if rollout else z[:, t:t + 1]
        with torch.no_grad():
            _denoise(dynamics, nxt, actions[t:t + 1], t, CONTEXT_T, generator, CONTEXT_STRIDE)
    return value


def clip_loss(dynamics, z, actions, event_at, rollout, generator):
    """Shortcut loss on frames CONTEXT..CLIP-1. Backward runs per frame, before the cache is
    overwritten. `rollout` writes the model's own prediction back, so later frames are trained
    under the drift that prediction causes; otherwise the real latent is written (teacher forcing)."""
    dynamics.clean_kvcache()
    with torch.no_grad():
        for i in range(CONTEXT):
            _denoise(dynamics, z[:, i:i + 1], actions[i:i + 1], i, CONTEXT_T, generator, CONTEXT_STRIDE)
    weights = [3.0 if event_at >= 0 and abs(t - event_at) <= 4 else 1.0 for t in range(CONTEXT, CLIP)]
    norm = sum(weights)
    value = 0.0
    for j, t in enumerate(range(CONTEXT, CLIP)):
        t_index = TRAIN_T[int(torch.randint(len(TRAIN_T), (), device=z.device, generator=generator))]
        pred = _denoise(dynamics, z[:, t:t + 1], actions[t - 1:t], t, t_index, generator)
        loss = weights[j] / norm * F.mse_loss(pred.float(), z[:, t:t + 1].float())
        loss.backward()
        value += float(loss.detach())
        nxt = pred.detach() if rollout else z[:, t:t + 1]
        with torch.no_grad():
            _denoise(dynamics, nxt, actions[t:t + 1], t, CONTEXT_T, generator, CONTEXT_STRIDE)
    return value


def _read_rows(path, start, count):
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= start + count:
                break
            rows.append(json.loads(line))
    return rows if len(rows) == count else None


def _read_frames(video, start, count):
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (360, 640):
            rgb = cv2.resize(rgb, (640, 360))
        frames.append(rgb)
    cap.release()
    return np.stack(frames) if len(frames) == count else None


def _tokens(codec, spec, rows):
    out = []
    for row in rows:
        env, _ = codec.json_action_to_env_action(row, hotbar=True)
        tok = np.asarray(codec.get_action_index_from_actiondict(env, include_gui=True), dtype=np.int64)
        if tok.shape != (len(spec.slots),) or any(tok[i] not in spec.slots[i] for i in range(len(spec.slots))):
            return None
        out.append(tok)
    return np.stack(out)


def _acquire_reader():
    locks = []
    for i in range(2):
        f = open(f'/dev/shm/v7_wm_ft_reader{i}.lock', 'a')
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except BlockingIOError:
            locks.append(f)
    for f in locks:
        f.close()
    f = open('/dev/shm/v7_wm_ft_reader0.lock', 'a')
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    _, _, spec, codec = _codec_only()
    jobs = _clip_jobs(args.output / 'probe')
    print(f'{len(jobs)} clips', flush=True)
    import multiprocessing as mp
    done = 0
    with mp.Pool(4) as pool:
        for name in pool.imap_unordered(_extract, [(j, str(args.output)) for j in jobs], chunksize=4):
            done += 1
            if done % 100 == 0:
                print(f'extracted {done}/{len(jobs)} {name}', flush=True)
    (args.output / 'prepare.json').write_text(json.dumps(dict(clips=done, code_sha256=sha256(__file__)), indent=1))
    print('prepare done', done)


def _codec_only():
    """Tokenizer is not needed to tokenize actions; still checks the asset hash via load path pieces."""
    assets = json.loads((ROOT / 'outputs/acceptance_20260913/assets.json').read_text())
    for entry in assets['files']:
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('WM assets changed')
    sys.path.insert(0, str(ROOT / 'Dreamerv4-MC'))
    from src.modules.actokenizer import MineCraftActionTokenizer
    spec = ActionSpec.from_dict(json.loads((ROOT / 'outputs/acceptance_20260913/action_probe.json').read_text())['action_spec'])
    return None, None, spec, MineCraftActionTokenizer()


def _clip_jobs(probe_dir):
    rng = np.random.default_rng(0)
    counts = {}
    jobs = []
    entries = [json.loads(l) for l in (CACHE / 'selection.jsonl').read_text().splitlines()]
    entries.sort(key=lambda e: session_group(e['session']) != 'select')  # hold out select diamond first
    for e in entries:
        group = session_group(e['session'])
        if group == 'confirm':
            continue
        frames = {m: [] for m in CAPS if m != 'general'}
        for ev in e.get('events', []):
            m = MILESTONE_TASK.get(ev['task'])
            if m:
                frames[m].append(int(ev['frame']))
        for m, fs in frames.items():
            split = 'probe' if group == 'select' and m == 'diamond' else 'train'
            cap = PROBE_CAP if split == 'probe' else (CAPS[m] if group == 'fit' else 0)
            key = (split, m)
            if counts.get(key, 0) >= cap or not fs:
                continue
            picks = fs if len(fs) <= 3 else list(rng.choice(fs, 3, replace=False))
            for frame in picks:
                if counts.get(key, 0) >= cap or frame < EVENT_INDEX or frame + (CLIP - EVENT_INDEX) > e['frames']:
                    continue
                jobs.append(dict(video=e['video'], actions=e['actions'], start=int(frame - EVENT_INDEX),
                                 skill=m, event_at=EVENT_INDEX, split=split))
                counts[key] = counts.get(key, 0) + 1
        if group == 'fit' and counts.get(('train', 'general'), 0) < CAPS['general']:
            start = int(rng.integers(0, max(1, e['frames'] - CLIP)))
            jobs.append(dict(video=e['video'], actions=e['actions'], start=start, skill='general', event_at=-1, split='train'))
            counts[('train', 'general')] = counts.get(('train', 'general'), 0) + 1
    return jobs


def _probe_jobs():
    """Held-out 'select' clips for drift probing: CONTEXT real frames, then PROBE_HORIZON frames replayed
    with the real actions; the event sits EVENT_INDEX - CONTEXT frames into the generated part."""
    counts, jobs = {}, []
    length = CONTEXT + PROBE_HORIZON
    for e in (json.loads(l) for l in (CACHE / 'selection.jsonl').read_text().splitlines()):
        if session_group(e['session']) != 'select':
            continue
        for ev in e.get('events', []):
            m = MILESTONE_TASK.get(ev['task'])
            frame = int(ev['frame'])
            if m not in PROBE_CAPS or counts.get(m, 0) >= PROBE_CAPS[m]:
                continue
            if frame < EVENT_INDEX or frame - EVENT_INDEX + length > e['frames']:
                continue
            if any(j['video'] == e['video'] and abs(j['start'] - (frame - EVENT_INDEX)) < length for j in jobs):
                continue
            jobs.append(dict(video=e['video'], actions=e['actions'], start=frame - EVENT_INDEX, skill=m,
                             event_at=EVENT_INDEX, split='probe_long', length=length))
            counts[m] = counts.get(m, 0) + 1
    return jobs


def prepare_probe(args):
    _codec_only()
    jobs = _probe_jobs()
    names = [_extract((j, str(args.output))) for j in jobs]
    print(f"probe_long: {sum(n != 'skip' for n in names)}/{len(jobs)} clips", flush=True)


_WORKER = {}


def _extract(payload):
    job, output = payload
    if not _WORKER:
        _, _, spec, codec = _codec_only()
        _WORKER['spec'], _WORKER['codec'] = spec, codec
    lock = _acquire_reader()
    try:
        length = job.get('length', CLIP)
        frames = _read_frames(job['video'], job['start'], length)
        rows = _read_rows(Path(job['actions']), job['start'], length) if frames is not None else None
    finally:
        lock.close()
    if frames is None or rows is None:
        return 'skip'
    actions = _tokens(_WORKER['codec'], _WORKER['spec'], rows)
    if actions is None:
        return 'skip'
    dest = Path(output) / job['split']
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{job['skill']}__{job['start']}__{Path(job['video']).stem}.npz"
    np.savez_compressed(dest / name, frames=frames, actions=actions.astype(np.int16),
                        event_at=job['event_at'], skill=job['skill'])
    return name


def _clips(root, split):
    return sorted((Path(root) / split).glob('*.npz'))


def training_objective(name):
    if name == 'shortcut_v2':
        return clip_loss_v2
    if name == 'legacy_denoise':
        return clip_loss
    raise ValueError('Unknown WM objective: ' + str(name))


def validate_resume_objective(config, requested):
    saved = config.get('objective')
    saved = {'v2': 'shortcut_v2'}.get(saved, saved)
    if saved is None:
        raise ValueError('Checkpoint lacks objective provenance; audit it before creating a new initialization run')
    if saved != requested:
        raise ValueError(f'Cannot resume objective {saved} as {requested}')


def train(args):
    import os
    import torch.distributed as dist
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    if world > 1:
        import datetime
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=15))
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        device = f"cuda:{os.environ['LOCAL_RANK']}"
    else:
        device = args.device
    objective = training_objective(args.objective)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    tokenizer, dynamics, spec, _ = load_backends(device)
    install_lora(dynamics, args.last_n, args.rank, args.alpha)
    params = [p for p in dynamics.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    first_step = 1
    if args.resume:
        blob = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_resume_objective(blob['config'], args.objective)
        load_lora_state(dynamics, blob['lora'])
        if 'opt' in blob:
            opt.load_state_dict(blob['opt'])
        first_step = int(blob['step']) + 1
    paths = _clips(args.clips, 'train')
    if not paths:
        raise SystemExit(f'no clips in {args.clips}/train')
    by = {}
    for p in paths:
        by.setdefault(p.name.split('__', 1)[0], []).append(p)
    skills = [s for s in by if s in WEIGHT]
    weights = np.array([WEIGHT[s] for s in skills], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng([args.seed, rank])
    generator = torch.Generator(device=device).manual_seed(args.seed * 1000 + rank)
    config = dict(clips=str(args.clips), n_clips=len(paths), per_skill={s: len(v) for s, v in by.items()},
                  last_n=args.last_n, rank=args.rank, alpha=args.alpha, lr=args.lr, steps=args.steps, world=world,
                  objective=args.objective, objective_function=objective.__name__,
                  clips_per_step=world, trainable=sum(p.numel() for p in params), code_sha256=sha256(__file__))
    if rank == 0:
        log = (args.output / 'metrics.jsonl').open('a' if args.resume else 'w')
        (args.output / ('config_resume.json' if args.resume else 'config.json')).write_text(json.dumps(dict(config, resume=str(args.resume) if args.resume else None), indent=1))
        print(json.dumps(dict(clips=len(paths), trainable=config['trainable'], world=world)), flush=True)
    started = time.time()
    for step in range(first_step, args.steps + 1):
        skill = skills[int(rng.choice(len(skills), p=weights))]
        path = by[skill][int(rng.integers(len(by[skill])))]
        with np.load(path) as z:
            frames, actions, event_at = z['frames'], z['actions'].astype(np.int64), int(z['event_at'])
        z_lat = _encode(tokenizer, frames, device)
        act = torch.tensor(actions, dtype=torch.long, device=device)
        rollout = skill in RARE
        opt.zero_grad(set_to_none=True)
        loss = objective(dynamics, z_lat, act, event_at, rollout, generator)
        if world > 1:
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().reshape(-1) for p in params]
                             + [torch.tensor([loss], device=device)])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            offset = 0
            for p in params:
                p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
                offset += p.numel()
            loss = float(flat[-1])
        if not np.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError('Nonfinite WM update; checkpoint not saved')
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        if step % 20 == 0 and rank == 0:
            rec = dict(step=step, skill=skill, rollout=rollout, loss=float(loss), sec=round(time.time() - started))
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(rec, flush=True)
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            torch.save(dict(lora=lora_state(dynamics), opt=opt.state_dict(), config=config, step=step), args.output / 'lora.pt')
    print('train done', flush=True)
    if world > 1:
        dist.destroy_process_group()


def cyan_fraction(frames):
    """Diamond-ore colour fraction per frame, [T,360,640,3] uint8 RGB; hotbar rows excluded."""
    import cv2
    out = []
    for f in frames:
        hsv = cv2.cvtColor(np.ascontiguousarray(f[:300]), cv2.COLOR_RGB2HSV).astype(np.float32)
        h, sat, val = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
        out.append(float(((h >= 160) & (h <= 200) & (sat > 0.4) & (val > 0.15)).mean()))
    return np.array(out)


def probe(args):
    from .world_v7 import AlignedWorldAdapter
    sys.path.insert(0, str(ROOT / 'Dreamerv4-MC'))
    from src.inference.infer import generate_one_frame
    tokenizer, dynamics, spec, _ = load_backends(args.device)
    if args.lora:
        blob = torch.load(args.lora, map_location='cpu', weights_only=False)
        install_lora(dynamics, blob['config']['last_n'], blob['config']['rank'], blob['config']['alpha'])
        load_lora_state(dynamics, blob['lora'])
    dynamics.eval()
    world = AlignedWorldAdapter(tokenizer, dynamics, generate_one_frame, spec, seed=0, steps=4)
    scores = []
    for path in _clips(args.clips, args.split)[:args.n]:
        with np.load(path) as z:
            frames, actions = z['frames'], torch.tensor(z['actions'].astype(np.int64), device=args.device)
            skill = str(z['skill'])
        pixels = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        pixels = F.pad(pixels, (0, 0, 12, 12))[None].to(args.device)
        for seed in range(args.seeds):
            world.reset(pixels[:, :CONTEXT], actions[:CONTEXT], seed=seed)
            err, gen = [], []
            for t in range(CONTEXT - 1, len(frames) - 1):
                _, pred = world.step(actions[t:t + 1])
                got = pred[0, 0, :, 12:372].float()
                err.append(float((got - pixels[0, t + 1, :, 12:372]).abs().mean()))
                gen.append(((got.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
            real_c, gen_c = cyan_fraction(frames[CONTEXT:]), cyan_fraction(np.stack(gen))
            visible = real_c > 5e-4
            scores.append(dict(clip=path.name, skill=skill, seed=seed, l1=err,
                               cyan_real=float(real_c[visible].sum()), cyan_gen=float(gen_c[visible].sum()),
                               cyan_gen_all=float(gen_c.sum()), cyan_real_all=float(real_c.sum())))
            print(dict(clip=path.name[:40], seed=seed, l1_first16=round(float(np.mean(err[:16])), 4),
                       l1_last32=round(float(np.mean(err[-32:])), 4),
                       cyan_ret=round(scores[-1]['cyan_gen'] / max(1e-9, scores[-1]['cyan_real']), 3)), flush=True)
    summary = dict(n=len(scores), lora=str(args.lora), split=args.split, per_skill={})
    for skill in sorted({s['skill'] for s in scores}):
        rows = [s for s in scores if s['skill'] == skill]
        curve = np.mean([s['l1'] for s in rows], 0)
        summary['per_skill'][skill] = dict(
            episodes=len(rows),
            l1_at={h: float(curve[:h].mean()) for h in (8, 16, 32, 64, 96) if h <= len(curve)},
            cyan_retention=sum(s['cyan_gen'] for s in rows) / max(1e-9, sum(s['cyan_real'] for s in rows)),
            cyan_gen_over_real_all=sum(s['cyan_gen_all'] for s in rows) / max(1e-9, sum(s['cyan_real_all'] for s in rows)))
    summary['clips'] = scores
    out = Path(args.report) if args.report else Path(args.clips) / ('probe_lora.json' if args.lora else 'probe_base.json')
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['per_skill'], indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('prepare-probe')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('train')
    p.add_argument('--clips', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--last-n', type=int, default=4)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=float, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--objective', choices=('shortcut_v2', 'legacy_denoise'), default='shortcut_v2')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('probe')
    p.add_argument('--clips', type=Path, required=True)
    p.add_argument('--lora', type=Path, default=None)
    p.add_argument('--n', type=int, default=36)
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--split', default='probe_long')
    p.add_argument('--report', type=Path, default=None)
    p.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    {'prepare': prepare, 'prepare-probe': prepare_probe, 'train': train, 'probe': probe}[args.cmd](args)


if __name__ == '__main__':
    main()
