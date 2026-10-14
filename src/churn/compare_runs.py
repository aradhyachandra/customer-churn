"""Compare two results/ folders file by file (used by the clean-clone check, docs/C6_reproducibility.md).

Run with:  python -m churn.compare_runs <first results dir> <second results dir> [--out file.json]

JSON files are compared after dropping run information (generated_at, validation_generated_at, git)
at any depth; every other file must be byte-identical.
"""

import argparse
import hashlib
import json
from pathlib import Path

RUN_INFO_KEYS = {"generated_at", "validation_generated_at", "git"}


def strip_run_info(obj):
    if isinstance(obj, dict):
        return {k: strip_run_info(v) for k, v in obj.items() if k not in RUN_INFO_KEYS}
    if isinstance(obj, list):
        return [strip_run_info(v) for v in obj]
    return obj


def _files(root):
    return {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file() and p.name != ".DS_Store"}


def compare_dirs(first, second):
    first, second = Path(first), Path(second)
    a, b = _files(first), _files(second)
    rows = []
    for name in sorted(a | b):
        if name not in b:
            status = "MISSING_IN_SECOND"
        elif name not in a:
            status = "MISSING_IN_FIRST"
        else:
            x, y = (first / name).read_bytes(), (second / name).read_bytes()
            if x == y:
                status = "identical"
            elif name.endswith(".json") and strip_run_info(json.loads(x)) == strip_run_info(json.loads(y)):
                status = "same_ignoring_run_info"
            else:
                status = "DIFFERENT"
        rows.append({"file": name, "status": status,
                     "sha256_first": hashlib.sha256((first / name).read_bytes()).hexdigest() if name in a else None,
                     "sha256_second": hashlib.sha256((second / name).read_bytes()).hexdigest() if name in b else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = compare_dirs(args.first, args.second)
    for r in rows:
        print(f"{r['status']:<24} {r['file']}")
    bad = [r for r in rows if r["status"] not in ("identical", "same_ignoring_run_info")]
    print(f"{len(rows)} files, {len(rows) - len(bad)} match, {len(bad)} do not")
    if args.out:
        Path(args.out).write_text(json.dumps({"first": args.first, "second": args.second, "files": rows},
                                             indent=2) + "\n")


if __name__ == "__main__":
    main()
