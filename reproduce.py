"""
reproduce.py
------------
Single entry point to reproduce all 4 phases of the sepsis
falsification study on MIMIC-IV v3.1.

Usage:
    python reproduce.py --mimic_path /path/to/mimiciv/3.1
    python reproduce.py --mimic_path /path/to/mimiciv/3.1 --output results.json
    python reproduce.py --mimic_path /path/to/mimiciv/3.1 --cache_dir /tmp/cache
    python reproduce.py --mimic_path /path/to/mimiciv/3.1 --seed 42

Expected output matches published results:
    Phase 1: NOT CONFIRMED  (mean_jaccard=0.5124)
    Phase 2: NOT CONFIRMED  (auc_drop=0.0027)
    Phase 3: NOT CONFIRMED  (care_intensity_auc=0.6603)
    Phase 4: NOT CONFIRMED  (discriminator_auc=0.6330)
    Verdict: Hypothesis not supported (0/4 confirmed)

OSF Pre-Registration: https://osf.io/9tbjm
Primary Run ID:       run_20260314_153526_9ac64a5a
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

from cohort   import build_cohort
from labels   import compute_labels
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
)


def run(mimic_path: str, cache_dir: str, output: str, seed: int):

    np.random.seed(seed)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("  SEPSIS FALSIFICATION STUDY — REPRODUCTION SCRIPT")
    print("  OSF: https://osf.io/9tbjm")
    print(f"  MIMIC-IV path: {mimic_path}")
    print(f"  Cache dir:     {cache_dir}")
    print(f"  Seed:          {seed}")
    print("=" * 60)
    print()

    # ── Cohort ────────────────────────────────────────────────────────────────
    cohort = build_cohort(
        mimic_path,
        cache_path=os.path.join(cache_dir, "cohort.parquet"),
    )
    print()

    # ── Vitals & Labs ─────────────────────────────────────────────────────────
    vitals = extract_vitals(
        cohort, mimic_path,
        cache_path=os.path.join(cache_dir, "vitals.parquet"),
    )
    labs = extract_labs(
        cohort, mimic_path,
        cache_path=os.path.join(cache_dir, "labs.parquet"),
    )
    print()

    # ── Labels ────────────────────────────────────────────────────────────────
    cohort = compute_labels(
        cohort, vitals, labs, mimic_path,
        cache_path=os.path.join(cache_dir, "cohort_with_labels.parquet"),
    )
    print()

    # ── Features ──────────────────────────────────────────────────────────────
    feat = build_feature_matrix(
        cohort, vitals, labs, mimic_path,
        cache_path=os.path.join(cache_dir, "features.parquet"),
    )
    print()

    bio_feats  = get_biological_features(feat)
    care_feats = get_care_intensity_features(feat)
    all_feats  = bio_feats + care_feats

    labels_sep3 = cohort.set_index("stay_id")["sepsis3"].reindex(feat["stay_id"])

    print(f"Feature matrix: {len(feat):,} stays × {len(all_feats)} features")
    print(f"  Biological features:     {len(bio_feats)}")
    print(f"  Care-intensity features: {len(care_feats)}")
    print(f"  Sepsis-3 positive:       {labels_sep3.sum():,} ({labels_sep3.mean()*100:.1f}%)")
    print()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    p1 = phase1_ground_truth_stability(cohort)
    print()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    p2 = phase2_feature_dependence(
        feat, labels_sep3, bio_feats, all_feats
    )
    print()

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    p3 = phase3_care_intensity_universality(feat, labels_sep3, care_feats)
    print()

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    p4 = phase4_synthetic_validation(feat, labels_sep3, care_feats)
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    phases     = [p1, p2, p3, p4]
    n_confirmed = sum(p["confirmed"] for p in phases)

    if n_confirmed == 4:
        verdict = "Care-process leakage hypothesis confirmed"
    elif n_confirmed >= 3:
        verdict = "Mixed evidence — partial support"
    else:
        verdict = "Hypothesis not supported"

    results = {
        "osf_registration": "https://osf.io/9tbjm",
        "seed":             seed,
        "cohort_n":         int(len(cohort)),
        "phases_confirmed": n_confirmed,
        "verdict":          f"{verdict} ({n_confirmed}/4 confirmed)",
        "phase1": {
            "confirmed":    p1["confirmed"],
            "mean_jaccard": p1["metrics"]["mean_jaccard"],
            "threshold":    p1["threshold"],
        },
        "phase2": {
            "confirmed":  p2["confirmed"],
            "auc_drop":   p2["metrics"]["auc_drop"],
            "auc_full":   p2["metrics"]["auc_full"],
            "auc_bio":    p2["metrics"]["auc_bio_only"],
            "threshold":  p2["threshold"],
        },
        "phase3": {
            "confirmed":         p3["confirmed"],
            "care_intensity_auc": p3["metrics"]["care_intensity_auc"],
            "threshold":         p3["threshold"],
        },
        "phase4": {
            "confirmed":        p4["confirmed"],
            "discriminator_auc": p4["metrics"]["discriminator_auc"],
            "threshold":        p4["threshold"],
        },
    }

    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Cohort N:   {len(cohort):,}")
    print(f"  Phase 1:    {'✓ CONFIRMED' if p1['confirmed'] else '✗ NOT CONFIRMED'}"
          f"  (Jaccard={p1['metrics']['mean_jaccard']})")
    print(f"  Phase 2:    {'✓ CONFIRMED' if p2['confirmed'] else '✗ NOT CONFIRMED'}"
          f"  (AUC drop={p2['metrics']['auc_drop']})")
    print(f"  Phase 3:    {'✓ CONFIRMED' if p3['confirmed'] else '✗ NOT CONFIRMED'}"
          f"  (Care AUC={p3['metrics']['care_intensity_auc']})")
    print(f"  Phase 4:    {'✓ CONFIRMED' if p4['confirmed'] else '✗ NOT CONFIRMED'}"
          f"  (Disc AUC={p4['metrics']['discriminator_auc']})")
    print(f"  VERDICT:    {verdict} ({n_confirmed}/4 confirmed)")
    print("=" * 60)

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results written to: {output}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce sepsis falsification study results on MIMIC-IV v3.1"
    )
    parser.add_argument(
        "--mimic_path", required=True,
        help="Root path to MIMIC-IV v3.1 (contains hosp/ and icu/ subdirs)"
    )
    parser.add_argument(
        "--output", default="results.json",
        help="Output JSON file for results (default: results.json)"
    )
    parser.add_argument(
        "--cache_dir", default="cache",
        help="Directory for intermediate parquet cache files (default: cache/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42, matches published results)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.mimic_path):
        print(f"Error: MIMIC-IV path not found: {args.mimic_path}")
        sys.exit(1)

    run(args.mimic_path, args.cache_dir, args.output, args.seed)


if __name__ == "__main__":
    main()
