"""Read-only P0 probe of installed inference code, not a production adapter."""
from pathlib import Path
import os, sys, subprocess, json, time, traceback
from dreamer_agent.provenance import write_report, sha256

def main():
    out=Path("outputs/acceptance_20260913")
    usage=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    candidates=[tuple(map(int,row.split(","))) for row in usage.strip().splitlines()]
    candidates=[r for r in candidates if r[1]>24000 and r[2]<=20]
    if not candidates:
        write_report(out/"inference.json",dict(passed=False,stage="resource_check",reason="No GPU with >24GB free and <=20% utilization"))
        return
    gpu=min(candidates,key=lambda r:(r[2],-r[1]))[0]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    os.environ["HF_HUB_OFFLINE"]="1"
    import torch
    import cv2
    import numpy as np
    sys.path.insert(0,str(Path("Dreamerv4-MC").resolve()))
    from src.inference.mc_vw_infer import MCWorldModelInfer
    from src.modules.tokenizer import CausalTokenizer
    from src.modules.dynamic_model import DynamicModel
    from src.modules.actokenizer import MineCraftActionTokenizer
    assets=json.loads((out/"assets.json").read_text())
    assert all(r["cache_blob_integrity"] is True for r in assets["files"] if r["path"].endswith(".safetensors"))
    snapshot=Path(assets["files"][0]["path"]).parents[1]
    report=dict(passed=False,gpu=gpu,revision=assets["revision"],stage="loading",loading_info={},
                preprocessing="first 2 RGB frames, float [-1,1], zero pad 12 pixels top/bottom; diagnostic convention, not certified",
                p0_passed=False,code_sha256={str(p):sha256(p) for p in Path("Dreamerv4-MC/src").rglob("*.py")})
    class AuditedInfer(MCWorldModelInfer):
        def init_tokenizer(self,tokenizer_path,device,subfolder=None):
            self.tokenizer, info=CausalTokenizer.from_pretrained(tokenizer_path,subfolder=subfolder,
                local_files_only=True,torch_dtype=torch.bfloat16,output_loading_info=True)
            report["loading_info"]["tokenizer"]=info
            if any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
                raise RuntimeError("Tokenizer state mismatch")
            self.tokenizer=self.tokenizer.to(device).eval().requires_grad_(False)
        def init_dynamic_model(self,dynamic_model_path,device,subfolder=None):
            self.model,info=DynamicModel.from_pretrained(dynamic_model_path,subfolder=subfolder,
                local_files_only=True,torch_dtype=torch.bfloat16,output_loading_info=True)
            report["loading_info"]["dynamic"]=info
            if any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
                raise RuntimeError("Dynamics state mismatch")
            self.model=self.model.to(device).eval().requires_grad_(False)
    try:
        torch.manual_seed(17)
        torch.cuda.reset_peak_memory_stats()
        model=AuditedInfer(str(snapshot),str(snapshot),str(out),4,torch.device("cuda"),
              torch.bfloat16,torch.Generator(device="cuda").manual_seed(17),
              use_cuda_graph=False,refresh_kvcache=False,tokenizer_subfolder="tokenizer",dynamic_subfolder="dynamic")
        report["stage"]="real_context_prefill"
        source=Path("data/vpt/raw/videos/cheeky-cornflower-setter-1d6469cf1ed4-20220421-150148.mp4")
        cap=cv2.VideoCapture(str(source))
        frames=[]
        for i in range(2):
            ok,frame=cap.read()
            if not ok:raise RuntimeError("Missing source frame")
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release()
        pixels=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float()/127.5-1
        pixels=torch.nn.functional.pad(pixels,(0,0,12,12))[None].to("cuda")
        rows=[]
        with Path("data/vpt/raw/actions/"+source.stem+".jsonl").open() as f:
            for _ in range(3):rows.append(json.loads(next(f)))
        codec=MineCraftActionTokenizer()
        actions=torch.tensor([codec.get_action_index_from_actiondict(codec.json_action_to_env_action(r,hotbar=True)[0],include_gui=True)
                              for r in rows],device="cuda",dtype=torch.long)
        start=time.perf_counter()
        with torch.inference_mode():
            model.prefilling_kvcache(pixels,actions[:2])
        torch.cuda.synchronize()
        report["prefill_seconds"]=time.perf_counter()-start
        report["stage"]="one_step"
        start=time.perf_counter()
        with torch.inference_mode():
            frame=model.render_next_frame(actions[2:3])
        torch.cuda.synchronize()
        report["step_seconds"]=time.perf_counter()-start
        report["output_shape"]=list(frame.shape)
        report["finite_output"]=bool(torch.isfinite(frame).all())
        report["passed"]=report["finite_output"]
        report["stage"]="completed_smoke_only"
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        report["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        write_report(out/"inference.json",report)
        print(json.dumps({k:v for k,v in report.items() if k not in ("code_sha256","traceback")},default=str),flush=True)

if __name__=="__main__":
    main()
