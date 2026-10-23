"""Audit the installed WM action wire format; does not certify game semantics."""
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np
from .actions import ActionSpec
from .provenance import sha256, write_report


def probe(codec_path, frame_hz, action_repeat):
    codec_path = Path(codec_path).resolve()
    module_spec = importlib.util.spec_from_file_location("_audited_mc_codec", codec_path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    codec = module.MineCraftActionTokenizer()
    codec.make_action_vocab()
    vocab = codec.action_vocab
    # Explicit compatibility hypothesis for this installed codec; verified below.
    names = [
        ["<act_bos>"], [f"cam_0_{i}" for i in range(21)],
        [f"cam_1_{i}" for i in range(21)],
        ["<null_act>"] + [f"hotbar.{i}" for i in range(1, 10)],
        ["<null_act>", "forward", "back"], ["<null_act>", "left", "right"],
        ["<null_act>", "sprint", "sneak"], ["<null_act>", "use", "attack"],
        ["<null_act>", "jump"], ["<null_act>", "drop", "pickItem"],
        ["<null_act>", "inventory", "ESC"], ["<act_eos>"]]
    spec = ActionSpec(tuple(tuple(vocab[n] for n in s) for s in names),
                      sha256(codec_path), frame_hz, action_repeat)
    cases = [("noop", {})]
    for key in ["forward", "back", "left", "right", "sprint", "sneak", "use",
                "attack", "jump", "drop", "pickItem", "inventory", "ESC"] + [f"hotbar.{i}" for i in range(1, 10)]:
        cases.append((key, {key: 1}))
    for axis in range(2):
        for value in range(21):
            cam = codec.camera_quantizer.undiscretize(np.array([value if axis == 0 else 10, value if axis == 1 else 10]))
            cases.append((f"camera_{axis}_{value}", {"camera": cam}))
    rows = []
    for name, action in cases:
        tokens = codec.get_action_index_from_actiondict(action, include_gui=True)
        recovered = spec.to_tokens(spec.to_local(np.array(tokens)))
        if not np.array_equal(tokens, recovered):
            raise ValueError(f"Round trip failed: {name}")
        rows.append({"case": name, "tokens": tokens})
    # Independent assertions catch missing GUI, swapped directions and dropped input.
    for name, _ in cases[1:23]:
        expected_slot = next(i for i, choices in enumerate(names) if name in choices)
        row = next(r for r in rows if r["case"] == name)
        if row["tokens"][expected_slot] != vocab[name]:
            raise ValueError(f"Input not preserved: {name}")
    broken = module.MineCraftActionTokenizer()
    broken.get_action_index_from_actiondict({}, include_gui=False)
    mixed = broken.get_action_index_from_actiondict({}, include_gui=True)
    mixed_bug = len(mixed) != len(spec.slots) or mixed[-1] != vocab["<act_eos>"]
    return {"action_spec": spec.to_dict(), "spec_sha256": spec.fingerprint,
            "codec_path": str(codec_path), "cases": rows, "wire_roundtrip_passed": True,
            "mixed_gui_call_corrupts_format": mixed_bug, "environment_roundtrip_passed": False,
            "timebase_status": "declared; requires video/environment alignment verification",
            "p0_passed": False, "scope": "Installed codec compatibility only; fresh GUI-enabled instances required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--frame-hz", type=float, required=True)
    parser.add_argument("--action-repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.codec, args.frame_hz, args.action_repeat)
    write_report(args.output, report)
    print(f"wire cases={len(report['cases'])}; mixed-GUI bug={report['mixed_gui_call_corrupts_format']}; P0 remains pending")


if __name__ == "__main__":
    main()
