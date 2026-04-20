# Sepsis Falsification Study — Replication Package

This directory contains a single-script replication runner that
reproduces all four pre-registered phases of the Sepsis Falsification
Study against MIMIC-IV v3.1 using only the public modules shipped in
this repository.

> **Pre-registration:** OSF, March 11, 2026 — https://osf.io/9tbjm
> **Pre-specified thresholds (locked, not modifiable):**
> Phase 1 mean Jaccard < 0.50 · Phase 2 AUC drop > 0.15 · Phase 3
> care-intensity AUC > 0.70 · Phase 4 discriminator AUC < 0.60.

## Quickstart

```bash
pip install -r requirements.txt

python replication_runner.py \
    --dataset_path /path/to/mimic-iv-3.1/ \
    --output_dir ./replication_run_mimic_iv
```

The first run builds caches under `./replication_run_*/cache/` and
takes 45–90 minutes on a single GPU or 4–8 hours on CPU. Subsequent
runs reuse caches and complete in ~10 minutes.

Output:
- `replication_results.json` — full results in the published schema
- `replication_summary.md` — human-readable summary table

## Data setup

MIMIC-IV v3.1 requires credentialed access from PhysioNet:
https://physionet.org/content/mimiciv/3.1/

`--dataset_path` should point to the directory containing the `hosp/`
and `icu/` subdirectories of `.csv.gz` files.

## Other datasets (MIMIC-III, eICU-CRD, Challenge 2019)

The public modules shipped here (`cohort.py`, `labels.py`,
`features.py`) are MIMIC-IV specific. To replicate against the other
three datasets used in the paper, adapt those three modules to the
target schema per the paper Methods:

- **MIMIC-III v1.4** — https://physionet.org/content/mimiciii/1.4/
- **eICU-CRD v2.0** — https://physionet.org/content/eicu-crd/2.0/
  (note: infection proxy uses antibiotic treatment alone; blood
  culture is available for <1% of stays — Pollard et al. 2018)
- **PhysioNet/CinC Sepsis Challenge 2019** —
  https://physionet.org/content/challenge-2019/1.0.0/

The four-phase logic in `phases.py` is dataset-agnostic and does not
need to change.

## Output schema (`replication_results.json`)

```json
{
  "schema_version": "1.0",
  "tool": "replication_runner.py",
  "dataset_type": "mimic_iv",
  "dataset_label": "MIMIC-IV v3.1",
  "seed": 42,
  "phases": {
    "phase1": {"phase": 1, "name": "...", "confirmed": false,
               "metrics": {"jaccard_sep2_sep3": ..., ...}, ...},
    "phase2": {...},
    "phase3": {...},
    "phase4": {...}
  },
  "verdict": {"verdict": "HYPOTHESIS NOT SUPPORTED",
              "phases_confirmed": 0, "phases_total": 4},
  "preregistration": {"osf_registration": "March 11, 2026", ...}
}
```

## Verifying you reproduced our results

Library-version drift can shift AUCs by ±0.005; differences smaller
than that are expected. Reference point estimates from the paper
(MIMIC-IV v3.1, primary analysis):

| Phase | Metric | Reproduced | Threshold |
|---|---|---|---|
| 1 | mean pairwise Jaccard | 0.5124 | < 0.50 → confirmed |
| 2 | AUC drop | 0.0027 | > 0.15 → confirmed |
| 3 | care-intensity AUC | 0.6603 | > 0.70 → confirmed |
| 4 | discriminator AUC | 0.6330 | < 0.60 → confirmed |

Published verdict: hypothesis not supported (0/4 confirmed).

## What is and isn't in this script

**Included (pre-registered confirmatory analysis):**
- Phase 1: Jaccard similarity across Sepsis-2, Sepsis-3, CMS SEP-1
- Phase 2: AUC drop after removing care-intensity features
- Phase 3: Logistic regression on care-intensity features only
- Phase 4: Gaussian copula synthetic vs real discriminator

**NOT included (post-hoc exploratory):**
- Phase 5 temporal drift / institution stratification / calibration
- SHAP and permutation-importance sensitivity analyses

These exploratory analyses are kept out of the replication runner to
preserve the pre-registered scope.

## Requirements

See `requirements.txt`. XGBoost auto-detects CUDA; CPU fallback is
automatic.

## Citation

If you use this replication package, please cite the paper:

> Dickens, A. (2026). Falsification Testing of Sepsis Prediction
> Models: Evaluating Independent Biological Signal After Controlling
> for Care-Process Intensity. *medRxiv* 2026.03.17.26348414v2.
> https://www.medrxiv.org/content/10.64898/2026.03.17.26348414v2

OSF pre-registration: https://osf.io/9tbjm

## Contact

Issues welcome at:
https://github.com/rocsite-research/sepsis-falsification
