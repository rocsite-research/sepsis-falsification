#!/usr/bin/env python3
"""
replication_runner.py
━━━━━━━━━━━━━━━━━━━━━
Standalone runner for the Sepsis Falsification Study.

Reproduces all four pre-registered phases against MIMIC-IV v3.1 using
only the public modules shipped in this repository (cohort.py,
labels.py, features.py, phases.py, utils.py). No additional pipeline
infrastructure is required.

Output JSON matches the schema published in the paper so independent
researchers can verify our results (allow ±0.005 tolerance on AUCs
for library-version drift).

Usage:
    python replication_runner.py \\
        --dataset_path /path/to/mimic-iv-3.1/ \\
        --output_dir ./replication_run

Pre-registration: OSF, March 11, 2026 — https://osf.io/9tbjm
Phase thresholds (locked):
    Phase 1: Mean pairwise Jaccard < 0.50
    Phase 2: AUC drop > 0.15
    Phase 3: Care-intensity AUC > 0.70
    Phase 4: Discriminator AUC < 0.60

Scope note:
    The public modules in this repository implement MIMIC-IV only. To
    replicate on MIMIC-III, eICU-CRD, or the PhysioNet/CinC 2019
    Challenge data, adapt cohort.py / labels.py / features.py to the
    target schema per the paper Methods, then re-run this script.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# Make the repo root importable so the public modules resolve even when
# this script is executed from inside replication/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cohort import build_cohort
from labels import compute_labels
from features import (
    extract_vitals,
    extract_labs,
    build_feature_matrix,
    get_biological_features,
    get_care_intensity_features,
)
from phases import (
    phase1_ground_truth_stability,
    phase2_feature_dependence,
    phase3_care_intensity_universality,
    phase4_synthetic_validation,
    THRESHOLD_P1_JACCARD,
    THRESHOLD_P2_AUC_DROP,
    THRESHOLD_P3_CARE_AUC,
    THRESHOLD_P4_DISC_AUC,
)


PHASE_THRESHOLDS = {
    1: ("Ground Truth Stability",      THRESHOLD_P1_JACCARD, "below"),
    2: ("Feature Dependence Test",     THRESHOLD_P2_AUC_DROP, "above"),
    3: ("Care-Intensity Universality", THRESHOLD_P3_CARE_AUC, "above"),
    4: ("Synthetic Validation",        THRESHOLD_P4_DISC_AUC, "below"),
}

DATASET_LABEL = "MIMIC-IV v3.1"


def _build_verdict(phases: list) -> dict:
    confirmed = [p for p in phases if p.get("confirmed")]
    n = len(confirmed)
    if n == 4:
        v = "HYPOTHESIS SUPPORTED"
        desc = "All 4 phases confirmed."
    elif n >= 2:
        v = "PARTIAL SUPPORT"
        desc = f"{n}/4 phases confirmed."
    else:
        v = "HYPOTHESIS NOT SUPPORTED"
        desc = (f"Only {n}/4 phases confirmed. Insufficient evidence for "
                "care-process leakage.")
    return {
        "verdict": v,
        "description": desc,
        "phases_confirmed": n,
        "phases_total": 4,
    }


def _run_phases(feat, cohort, labels_sep3, bio_feats, care_feats,
                all_feats, skip_phase):
    phases, errors = [], []

    if 1 not in skip_phase:
        print("\n[Phase 1] Ground Truth Stability (threshold below 0.50)")
        try:
            phases.append(phase1_ground_truth_stability(cohort))
        except Exception as e:
            errors.append({"phase": 1, "error": str(e),
                           "traceback": traceback.format_exc()})
            print(f"  → FAILED: {e}")

    if 2 not in skip_phase:
        print("\n[Phase 2] Feature Dependence Test (threshold above 0.15)")
        try:
            phases.append(phase2_feature_dependence(
                feat, labels_sep3, bio_feats, all_feats
            ))
        except Exception as e:
            errors.append({"phase": 2, "error": str(e),
                           "traceback": traceback.format_exc()})
            print(f"  → FAILED: {e}")

    if 3 not in skip_phase:
        print("\n[Phase 3] Care-Intensity Universality (threshold above 0.70)")
        try:
            phases.append(phase3_care_intensity_universality(
                feat, labels_sep3, care_feats
            ))
        except Exception as e:
            errors.append({"phase": 3, "error": str(e),
                           "traceback": traceback.format_exc()})
            print(f"  → FAILED: {e}")

    if 4 not in skip_phase:
        print("\n[Phase 4] Synthetic Validation (threshold below 0.60)")
        try:
            phases.append(phase4_synthetic_validation(
                feat, labels_sep3, care_feats
            ))
        except Exception as e:
            errors.append({"phase": 4, "error": str(e),
                           "traceback": traceback.format_exc()})
            print(f"  → FAILED: {e}")

    return phases, errors


def main():
    ap = argparse.ArgumentParser(
        description=("Replicate the Sepsis Falsification Study (4 phases) "
                     "against your local copy of MIMIC-IV v3.1.")
    )
    ap.add_argument("--dataset_path", required=True,
                    help="Root path of MIMIC-IV v3.1 (contains hosp/ and "
                         "icu/ subdirectories).")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write replication_results.json, "
                         "replication_summary.md, and pipeline cache.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42, matches published "
                         "results).")
    ap.add_argument("--skip_phase", type=int, action="append", default=[],
                    help="Skip phase N (e.g. --skip_phase 4 to skip the "
                         "expensive synthetic-validation phase). Repeatable.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Sepsis Falsification — Replication Runner")
    print(f"  dataset      : {DATASET_LABEL}")
    print(f"  dataset_path : {args.dataset_path}")
    print(f"  output_dir   : {out_dir}")
    print(f"  seed         : {args.seed}  (matches paper)")
    print("=" * 72)

    if not os.path.exists(args.dataset_path):
        print(f"ERROR: dataset_path does not exist: {args.dataset_path}")
        sys.exit(2)

    np.random.seed(args.seed)
    t0 = time.time()

    # Cohort → labels → features, reusing the public MIMIC-IV modules.
    cohort = build_cohort(
        args.dataset_path,
        cache_path=str(cache_dir / "cohort.parquet"),
    )
    vitals = extract_vitals(
        cohort, args.dataset_path,
        cache_path=str(cache_dir / "vitals.parquet"),
    )
    labs = extract_labs(
        cohort, args.dataset_path,
        cache_path=str(cache_dir / "labs.parquet"),
    )
    cohort = compute_labels(
        cohort, vitals, labs, args.dataset_path,
        cache_path=str(cache_dir / "cohort_with_labels.parquet"),
    )
    feat = build_feature_matrix(
        cohort, vitals, labs, args.dataset_path,
        cache_path=str(cache_dir / "features.parquet"),
    )
    bio_feats  = get_biological_features(feat)
    care_feats = get_care_intensity_features(feat)
    all_feats  = bio_feats + care_feats

    labels_sep3 = (
        cohort.set_index("stay_id")["sepsis3"].reindex(feat["stay_id"])
    )

    print(f"\nFeature matrix: {len(feat):,} stays × {len(all_feats)} features")
    print(f"  Biological features:     {len(bio_feats)}")
    print(f"  Care-intensity features: {len(care_feats)}")
    print(f"  Sepsis-3 positive:       {int(labels_sep3.sum()):,} "
          f"({labels_sep3.mean()*100:.1f}%)")

    phases, errors = _run_phases(
        feat, cohort, labels_sep3, bio_feats, care_feats, all_feats,
        set(args.skip_phase),
    )

    verdict = _build_verdict(phases)

    summary = {
        "schema_version": "1.0",
        "tool": "replication_runner.py",
        "tool_version": "1.0.0",
        "dataset_type": "mimic_iv",
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "dataset_label": DATASET_LABEL,
        "seed": args.seed,
        "cohort_n": int(len(cohort)),
        "timestamp_start": datetime.fromtimestamp(t0).isoformat(),
        "timestamp_end": datetime.now().isoformat(),
        "duration_seconds": round(time.time() - t0, 2),
        "phases": {f"phase{p['phase']}": p for p in phases},
        "verdict": verdict,
        "errors": errors,
        "preregistration": {
            "osf_registration": "March 11, 2026",
            "osf_url": "https://osf.io/9tbjm",
            "thresholds": {
                f"phase{k}": {"threshold": v[1], "direction": v[2]}
                for k, v in PHASE_THRESHOLDS.items()
            },
        },
    }

    json_path = out_dir / "replication_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {json_path}")

    md = [
        "# Replication Run Summary",
        f"- Dataset: **{DATASET_LABEL}** (`mimic_iv`)",
        f"- Cohort N: {len(cohort):,}",
        f"- Seed: {args.seed}",
        f"- Duration: {summary['duration_seconds']}s",
        f"- Verdict: **{verdict['verdict']}** "
        f"({verdict['phases_confirmed']}/{verdict['phases_total']} confirmed)",
        "",
        "## Phase results",
        "",
        "| Phase | Name | Key metric | Threshold | Confirmed |",
        "|---|---|---|---|---|",
    ]
    for p in phases:
        n = p["phase"]
        name = p.get("name", "")
        metrics = p.get("metrics", {})
        metric_key = next(iter(metrics), "")
        metric_val = metrics.get(metric_key, "")
        thr = p.get("threshold", "")
        direction = PHASE_THRESHOLDS[n][2]
        md.append(
            f"| {n} | {name} | {metric_key}={metric_val} | "
            f"{direction} {thr} | "
            f"{'YES' if p.get('confirmed') else 'no'} |"
        )
    if errors:
        md += ["", "## Errors", ""]
        for e in errors:
            md.append(f"- Phase {e['phase']}: {e['error']}")
    md_path = out_dir / "replication_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {md_path}")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
