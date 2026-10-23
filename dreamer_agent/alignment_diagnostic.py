"""Scene-motion/action lag diagnostics; never auto-certify frame alignment."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .provenance import sha256,write_report


def lag_scores(motion,actions,valid,lags=tuple(range(-3,4))):
    motion,actions,valid=np.asarray(motion),np.asarray(actions),np.asarray(valid,dtype=bool)
    if motion.ndim!=1 or actions.ndim!=1 or valid.shape!=motion.shape or not lags:
        raise ValueError("Expected vectors and nonempty lags")
    indices=np.arange(len(motion))
    common=valid & np.isfinite(motion)
    for lag in lags:
        bounded=(indices+lag>=0)&(indices+lag<len(actions))
        finite=np.zeros(len(indices),dtype=bool)
        finite[bounded]=np.isfinite(actions[(indices+lag)[bounded]])
        common &= bounded & finite
    indices=indices[common]
    rows=[]
    for lag in lags:
        x,y=motion[indices],actions[indices+lag]
        corr=float(np.corrcoef(x,y)[0,1]) if len(x)>=20 and np.std(x)>1e-8 and np.std(y)>1e-8 else None
        rows.append(dict(lag=int(lag),n=len(x),correlation=corr))
    return rows


def diagnose(video,actions,max_frames=600):
    if max_frames<25:raise ValueError("Need at least 25 frames")
    rows=[]
    with Path(actions).open() as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows)>=max_frames:break
    if any(not isinstance(r.get("isGuiOpen"),bool) for r in rows):
        raise ValueError("Explicit GUI metadata required for diagnostic masking")
    mouse=np.array([float(r["mouse"]["dx"]) for r in rows])
    cap=cv2.VideoCapture(str(video))
    previous=None
    shifts,responses=[],[]
    window=cv2.createHanningWindow((240,144),cv2.CV_32F)
    try:
        while len(shifts)<min(max_frames-1,len(rows)-1):
            ok,frame=cap.read()
            if not ok:break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            h,w=gray.shape
            gray=cv2.resize(gray[h//6:4*h//5,w//8:7*w//8],(240,144)).astype(np.float32)
            if previous is not None:
                shift,response=cv2.phaseCorrelate(previous.copy(),gray.copy(),window)
                shifts.append(-shift[0])
                responses.append(response)
            previous=gray
    finally:cap.release()
    if not shifts:raise ValueError("No decodable transitions")
    n=len(shifts)
    valid=np.array([all(not rows[j]["isGuiOpen"] for j in range(max(0,i-3),min(len(rows),i+5))) for i in range(n)])
    valid &= np.array(responses)>.15
    valid &= np.abs(shifts)<60
    scores=lag_scores(shifts,mouse,valid)
    order=sorted([r for r in scores if r["correlation"] is not None],key=lambda r:r["correlation"],reverse=True)
    return dict(video=str(Path(video).resolve()),actions=str(Path(actions).resolve()),
        video_sha256=sha256(video),actions_sha256=sha256(actions),transitions=n,
        scores=scores,leading_lag=order[0]["lag"] if order else None,
        correlation_margin=order[0]["correlation"]-order[1]["correlation"] if len(order)>1 else None,
        lag_definition="frame[t]->frame[t+1] motion versus mouse.dx[t+lag]",alignment_verified=False,
        limitations="Translation proxy confounds rotation and player movement; serially correlated actions. Diagnostic only; no automatic cropping.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",type=Path,required=True)
    p.add_argument("--actions",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-frames",type=int,default=600)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=diagnose(a.video,a.actions,a.max_frames)
    write_report(a.output,r)
    print({k:r[k] for k in ("transitions","leading_lag","correlation_margin","alignment_verified")})


if __name__=="__main__":
    main()
