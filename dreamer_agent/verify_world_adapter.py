"""Real-weight regression for the new orchestration; not semantic P0 acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .provenance import sha256, write_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=4)
    parser.add_argument("--rollout",type=int,default=34)
    args=parser.parse_args()
    if args.rollout < 1 or args.steps not in (1,2,4,8,16,32):
        parser.error("Invalid rollout/sampling steps")
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,p0_passed=False,steps=args.steps,rollout=args.rollout,stage="assets")
    try:
        assets=json.loads(Path("outputs/acceptance_20260913/assets.json").read_text())
        verified=json.loads(Path("outputs/acceptance_20260913/asset_verification.json").read_text())
        if not verified["passed"] or verified["revision"]!=assets["revision"]:
            raise ValueError("Official asset verification missing")
        for item in assets["files"]:
            if sha256(item["path"])!=item["sha256"]:
                raise ValueError("Asset changed since verification")
        report["revision"]=assets["revision"]
        report["stage"]="resource_check"
        usage=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
        choices=[tuple(map(int,row.split(","))) for row in usage.strip().splitlines()]
        choices=[r for r in choices if r[1]>20000 and r[2]<=20]
        if not choices:raise RuntimeError("No GPU with >20GB free and <=20% utilization")
        gpu=min(choices,key=lambda r:(r[2],-r[1]))[0]
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
        os.environ["HF_HUB_OFFLINE"]="1"
        import torch
        import cv2
        import numpy as np
        from .world_model import FrozenWorldAdapter
        from .actions import ActionSpec
        report["gpu"]=gpu
        sys.path.insert(0,str(Path("Dreamerv4-MC").resolve()))
        from src.modules.tokenizer import CausalTokenizer
        from src.modules.dynamic_model import DynamicModel
        from src.modules.actokenizer import MineCraftActionTokenizer
        from src.inference.infer import generate_one_frame
        snapshot=Path(assets["files"][0]["path"]).parents[1]
        report["stage"]="loading"
        modules=[]
        report["loading_info"]={}
        for name,cls in (("tokenizer",CausalTokenizer),("dynamic",DynamicModel)):
            module,info=cls.from_pretrained(str(snapshot),subfolder=name,local_files_only=True,
                  torch_dtype=torch.bfloat16,output_loading_info=True)
            report["loading_info"][name]=info
            if any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
                raise ValueError("State mismatch: "+name)
            modules.append(module.to("cuda").eval().requires_grad_(False))
        spec_data=json.loads(Path("outputs/acceptance_20260913/action_probe.json").read_text())["action_spec"]
        codec_path=Path("Dreamerv4-MC/src/modules/actokenizer.py")
        if sha256(codec_path)!=spec_data["codec_sha256"]:raise ValueError("Codec changed")
        adapter=FrozenWorldAdapter(*modules,generate_one_frame,ActionSpec.from_dict(spec_data),steps=args.steps)
        source=Path("data/vpt/raw/videos/cheeky-cornflower-setter-1d6469cf1ed4-20220421-150148.mp4")
        cap=cv2.VideoCapture(str(source));frames=[]
        try:
            for _ in range(4):
                ok,frame=cap.read()
                if not ok:raise ValueError("Video missing a context frame")
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally:cap.release()
        pixels=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float()/127.5-1
        pixels=torch.nn.functional.pad(pixels,(0,0,12,12))[None].to("cuda")
        with Path("data/vpt/raw/actions",source.stem+".jsonl").open() as f:
            rows=[json.loads(next(f)) for _ in range(4)]
        codec=MineCraftActionTokenizer()
        actions=torch.tensor([codec.get_action_index_from_actiondict(codec.json_action_to_env_action(row,hotbar=True)[0],include_gui=True) for row in rows],device="cuda",dtype=torch.long)
        report["context_action_alignment"]="diagnostic same-index convention; real dataset offset not certified"
        report["preprocessing"]="RGB [-1,1], vertical padding 12+12; same convention as failing regression"
        report["stage"]="prefill_2"
        torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        adapter.reset(pixels[:,:2],actions[:2],seed=17)
        torch.cuda.synchronize()
        report["prefill_2_seconds"]=time.perf_counter()-t
        report["prefill_2_next_index"]=adapter.frame_index
        global_cuda=torch.cuda.get_rng_state().clone()
        root=adapter.clone_state()
        latent_a,frame_a=adapter.step(actions[2:3])
        adapter.restore_state(root)
        alternative=actions[2:3].clone()
        alternative[:,4]=codec.action_vocab["forward"]
        adapter.step(alternative)
        adapter.restore_state(root)
        replay,replay_frame=adapter.step(actions[2:3])
        report["branch_replay_exact"]=bool(torch.equal(latent_a,replay) and torch.equal(frame_a,replay_frame))
        report["branch_replay_max_abs"]=float((latent_a-replay).abs().max())
        report["generator_isolated"]=bool(torch.equal(global_cuda,torch.cuda.get_rng_state()))
        del root
        adapter.reset(pixels[:,:2],actions[:2],seed=17)
        repeated,repeated_frame=adapter.step(actions[2:3])
        report["reset_replay_exact"]=bool(torch.equal(latent_a,repeated) and torch.equal(frame_a,repeated_frame))
        report["stage"]="prefill_4"
        adapter.reset(pixels,actions,seed=17)
        report["prefill_4_next_index"]=adapter.frame_index
        report["stage"]="rollout"
        measurements=[]
        report["timing"]=measurements
        report["wrap_replay_exact"]=None
        for index in range(args.rollout):
            if adapter.frame_index == int(adapter.dynamics.config.window_size):
                boundary = adapter.clone_state()
                after_wrap, after_wrap_frame = adapter.step(actions[3:4])
                adapter.restore_state(boundary)
                adapter.step(alternative)
                adapter.restore_state(boundary)
                wrap_replay, wrap_frame = adapter.step(actions[3:4])
                report["wrap_replay_exact"] = bool(torch.equal(after_wrap, wrap_replay) and torch.equal(after_wrap_frame, wrap_frame))
                adapter.restore_state(boundary)
                del boundary, after_wrap, after_wrap_frame, wrap_replay, wrap_frame
            torch.cuda.synchronize();start=time.perf_counter()
            latent,frame=adapter.step(actions[3:4])
            torch.cuda.synchronize()
            measurements.append(dict(step=index+1,seconds=time.perf_counter()-start,
                                     allocated_bytes=torch.cuda.memory_allocated()))
            if (index+1) % 32 == 0:
                with (args.output/"progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(measurements[-1])+"\n")
                print("completed model steps:", index+1, flush=True)
        report["timing"]=measurements
        report["last_frame_shape"]=list(frame.shape)
        report["last_latent_shape"]=list(latent.shape)
        report["next_frame_index"]=adapter.frame_index
        report["frozen_parameters"]=all(not p.requires_grad for m in modules for p in m.parameters())
        report["passed"]=all(report[k] for k in ("branch_replay_exact","reset_replay_exact","generator_isolated","frozen_parameters"))
        if args.rollout + 4 > int(adapter.dynamics.config.window_size):
            report["passed"] = report["passed"] and report["wrap_replay_exact"] is True
        # Output is a debugging artifact, not success evidence.
        image=((frame[0,0].float().cpu().clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
        cv2.imwrite(str(args.output/"last_generated_frame.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        report["stage"]="completed_technical_regression"
        report["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        report["code_sha256"]={str(p):sha256(p) for p in [Path(__file__),Path("dreamer_agent/world_model.py")]}
        report["backend_code_sha256"]={str(p):sha256(p) for p in Path("Dreamerv4-MC/src").rglob("*.py")}
        write_report(args.output/"report.json",report)
        print(json.dumps({k:v for k,v in report.items() if k not in ("traceback","timing","code_sha256","backend_code_sha256")}),flush=True)
    if not report["passed"]:sys.exit(1)


if __name__=="__main__":
    main()
