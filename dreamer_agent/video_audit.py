"""P0 video/action length and timebase check; causal offset remains unverified."""
import argparse
import json
from fractions import Fraction
from pathlib import Path
import subprocess
from .provenance import sha256, write_report
from .vpt_audit import inspect


def inspect_pair(video, actions, expected_hz):
    if expected_hz <= 0:
        raise ValueError("Expected FPS must be positive")
    raw = inspect(actions)
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
         "-of", "json", str(video)], check=True, capture_output=True, text=True, timeout=60)
    streams = json.loads(completed.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("Expected exactly one selected video stream")
    stream = streams[0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    count = int(stream["nb_frames"]) if stream.get("nb_frames", "N/A") != "N/A" else None
    return dict(video=str(Path(video).resolve()), video_sha256=sha256(video),
                actions=raw, stream=stream,
                length_and_rate_match=bool(raw["eligible_for_alignment_review"] and
                    count == raw["valid_rows"] and abs(fps - expected_hz) < 1e-6),
                frame_minus_action_count=count-raw["valid_rows"] if count is not None else None,
                possible_extra_observation=count == raw["valid_rows"]+1 if count is not None else False,
                count_source="container metadata; not full decode validation",
                action_observation_offset_verified=False, p0_passed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--actions", type=Path, required=True)
    p.add_argument("--expected-hz", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = inspect_pair(args.video, args.actions, args.expected_hz)
    write_report(args.output, report)
    print("length/rate match:", report["length_and_rate_match"], "; causal alignment remains pending")


if __name__ == "__main__":
    main()
