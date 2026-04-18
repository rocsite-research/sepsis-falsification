# Sepsis Prediction Falsification Study
## Reproduction Code

*Dedicated to Tanya, who co-founded RocSite and believed in this work from the beginning.  
And to the memory of Liane's father, who died from sepsis. This is why it matters.*

---

**Paper:** [Falsification Testing of Sepsis Prediction Models: Evaluating Independent Biological Signal After Controlling for Care-Process Intensity](https://www.medrxiv.org/content/10.64898/2026.03.17.26348414v2)  
medRxiv 2026.03.17.26348414 · v2 (corrected)

**OSF Pre-Registration:** March 11, 2026 · https://osf.io/9tbjm  
**Primary Run ID:** run_20260314_153526_9ac64a5a  
**Investigator:** Adam Dickens

---

## Overview

This repository contains the minimal reproduction code for the statistical
analyses described in the paper. It does not contain the full analysis platform
used to generate the published results. The platform implementation is maintained
separately under applicable intellectual property protections. See the OSF
pre-registration for full methodology details.

Reviewers and researchers with MIMIC-IV access can use this code to independently
verify the published phase results.

---

## v2 Correction Note

`labels.py` was updated to fix the GCS computation:

- `charttime` was missing from `usecols`, causing silent `sofa_cns=0` for all stays
- Replaced exact timestamp grouping with `merge_asof` (5-minute tolerance) to correctly pair eye/verbal/motor GCS components charted at slightly different timestamps

Rerun confirmed results unchanged: Sepsis-3 count 20,780→20,764, mean Jaccard 0.5124→0.5122, 0/4 phases confirmed.

---

## Requirements

- MIMIC-IV Clinical Database v3.1 (PhysioNet access required)
  https://physionet.org/content/mimiciv/3.1/
- Python 3.10+
- See `requirements.txt`

---

## Usage

```bash
pip install -r requirements.txt

python reproduce.py --mimic_path /path/to/mimiciv/3.1 --output results.json
```

Expected runtime: 45-90 minutes on first run (feature extraction).
Subsequent runs use cached parquet files.

---

## Expected Output

```json
{
  "phase1": {"confirmed": false, "mean_jaccard": 0.5122, "threshold": 0.50},
  "phase2": {"confirmed": false, "auc_drop": 0.0033,   "threshold": 0.15},
  "phase3": {"confirmed": false, "care_auc": 0.6605,   "threshold": 0.70},
  "phase4": {"confirmed": false, "disc_auc": 0.6318,   "threshold": 0.60},
  "verdict": "Hypothesis not supported (0/4 confirmed)"
}
```

---

## Repository Structure

```
sepsis_falsification/
├── README.md         — this file
├── requirements.txt  — dependencies
├── reproduce.py      — single entry point, runs all 4 phases
├── cohort.py         — inclusion/exclusion criteria
├── labels.py         — sepsis-2, sepsis-3, CMS definitions (v2 corrected)
├── features.py       — biological and care-intensity features
├── phases.py         — 4 phase statistical computations
└── utils.py          — shared utilities (jaccard, bootstrap, etc.)
```

---

## Citation

> Dickens, A. (2026). Falsification Testing of Sepsis Prediction Models:
> Evaluating Independent Biological Signal After Controlling for Care-Process Intensity.
> *medRxiv* 2026.03.17.26348414v2. https://www.medrxiv.org/content/10.64898/2026.03.17.26348414v2

---

## License

Apache License 2.0. See LICENSE.

---

## Note on Scope

This code implements the statistical methodology described in the paper.
The production analysis platform — including multi-dataset orchestration,
audit trail generation, and run recording infrastructure — is not included
and is not required to reproduce the published MIMIC-IV results.