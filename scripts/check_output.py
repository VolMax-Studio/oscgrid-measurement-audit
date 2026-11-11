#!/usr/bin/env python3
"""Compare a recomputed audit JSON to the published pinned reference."""
import argparse
import json
import sys
from pathlib import Path

EXPECTED = Path(__file__).resolve().parents[1] / "reference_results" / "structural_audit_expected.json"


def differences(a, b, prefix=""):
    out = []
    for key in sorted(set(a) | set(b)):
        path = prefix + "." + key if prefix else key
        if key not in a or key not in b:
            out.append((path, a.get(key, "<absent>"), b.get(key, "<absent>")))
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            out.extend(differences(a[key], b[key], path))
        elif a[key] != b[key]:
            out.append((path, a[key], b[key]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("actual", type=Path)
    args = ap.parse_args()
    expected = json.loads(EXPECTED.read_text(encoding="utf8"))
    actual = json.loads(args.actual.read_text(encoding="utf8"))
    diff = differences(expected, actual)
    if diff:
        print(f"FAIL: {len(diff)} fields differ")
        for key, exp, got in diff[:30]:
            print(f"{key}: expected={exp!r}; actual={got!r}")
        sys.exit(1)
    print("PASS: all structural audit fields match the pinned independent reference")
