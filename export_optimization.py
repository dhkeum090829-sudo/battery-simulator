"""Save a reproducible candidate search, with optional paired stress validation."""

import argparse
from pathlib import Path
import io
import json
import zipfile
from battery_model import Config
from optimization import (
    SearchSpace,
    Limits,
    search,
    recommend,
    validate_candidate,
    export_search,
)


def export_optimization(folder, pairs=16, priority="plating", validate=False):
    folder.mkdir(parents=True, exist_ok=True)

    def progress(n, total, label):
        if n == total or n % 10 == 0:
            print(f"{n}/{total} {label}", flush=True)

    report = search(Config(), SearchSpace(pairs=pairs), Limits(), progress=progress)
    chosen = recommend(report, priority)
    checks = (
        validate_candidate(report, chosen, progress=progress)
        if validate and chosen
        else None
    )
    packet = export_search(report, priority, checks, chosen)
    (folder / "charging_optimization.zip").write_bytes(packet)
    with zipfile.ZipFile(io.BytesIO(packet)) as archive:
        for name in (
            "search.csv",
            "metadata.json",
            "recommended_control.json",
            "validation.csv",
            "MODEL_NOTES.md",
        ):
            if name in archive.namelist():
                (folder / name).write_bytes(archive.read(name))
    summary = {
        "candidate_count": len(report.table),
        "feasible_count": int(report.table.feasible.sum()),
        "pareto_count": int(report.table.pareto.sum()),
        "recommended": chosen,
        "validation_count": len(checks) if checks is not None else 0,
        "validation_passed": int(checks.feasible.sum()) if checks is not None else None,
    }
    if chosen:
        summary["candidate"] = (
            report.table.set_index("candidate_id").loc[chosen].to_dict()
        )
        summary["reference"] = report.baseline
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return report, checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument(
        "--priority", choices=["plating", "time", "energy"], default="plating"
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    export_optimization(args.output, args.pairs, args.priority, args.validate)
