"""Read raw VPT JSONL; inventory/coordinates stay diagnostic metadata only."""
import argparse
import json
import math
from pathlib import Path
from .provenance import sha256, write_report


def inspect(path):
    count, previous, gaps, inventory_rows, stats_rows = 0, None, 0, 0, 0
    errors = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row.get("keyboard", {}).get("keys"), list):
                    raise ValueError("missing keyboard.keys")
                mouse = row.get("mouse", {})
                if not all(k in mouse for k in ("dx", "dy", "buttons")):
                    raise ValueError("missing mouse deltas/buttons; cannot assume no-op")
                if not all(type(mouse[k]) in (int, float) and math.isfinite(mouse[k]) for k in ("dx", "dy")) or not isinstance(mouse["buttons"], list):
                    raise ValueError("invalid mouse values")
                tick = row.get("tick")
                if type(tick) is not int:
                    raise ValueError("missing integer tick")
                if previous is not None:
                    if tick <= previous:
                        raise ValueError("nonmonotonic tick")
                    gaps += int(tick != previous + 1)
                previous = tick
                inventory_rows += int("inventory" in row)
                stats_rows += int(bool(row.get("stats")))
                count += 1
            except (ValueError, TypeError, AttributeError) as exc:
                if len(errors) < 10:
                    errors.append({"line": line_number, "reason": str(exc)})
    return {"path": str(Path(path).resolve()), "sha256": sha256(path),
            "valid_rows": count, "tick_gaps": gaps, "errors": errors,
            "inventory_field_rows": inventory_rows, "nonempty_stats_rows": stats_rows,
            "eligible_for_alignment_review": count > 0 and not errors and gaps == 0,
            "video_alignment_verified": False, "event_labels_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema_version": 1, "episodes": [inspect(p) for p in args.paths],
              "scope": "Raw metadata diagnostics, not training provenance certification"}
    write_report(args.output, report)
    print(f"audited {len(report['episodes'])} JSONL files")


if __name__ == "__main__":
    main()
