"""
cohort.py
---------
MIMIC-IV cohort construction: inclusion/exclusion criteria.

Inclusion:
  - Adult patients (age >= 18)
  - ICU LOS >= 4 hours
  - First ICU stay per patient only

Returns a DataFrame with columns:
  stay_id, subject_id, hadm_id, age, gender, intime, outtime, los_hours
"""

import os
import pandas as pd


def build_cohort(mimic_path: str, cache_path: str = None) -> pd.DataFrame:
    """
    Build the analytic cohort from MIMIC-IV.

    Parameters
    ----------
    mimic_path : str
        Root path to MIMIC-IV v3.1 (contains hosp/ and icu/ subdirs)
    cache_path : str, optional
        If provided, cache the cohort parquet here for faster re-runs

    Returns
    -------
    pd.DataFrame
        Final analytic cohort
    """
    if cache_path and os.path.exists(cache_path):
        print("Cohort: loading from cache")
        return pd.read_parquet(cache_path)

    print("Cohort: building from MIMIC-IV...")

    icu_path  = os.path.join(mimic_path, "icu", "icustays.csv.gz")
    adm_path  = os.path.join(mimic_path, "hosp", "admissions.csv.gz")
    pat_path  = os.path.join(mimic_path, "hosp", "patients.csv.gz")

    icu = pd.read_csv(icu_path, parse_dates=["intime", "outtime"])
    adm = pd.read_csv(adm_path, parse_dates=["admittime", "dischtime"])
    pat = pd.read_csv(pat_path)

    # Compute age at ICU admission
    # MIMIC-IV anchor_age + years since anchor_year
    pat["dob_proxy"] = pd.to_datetime(
        pat["anchor_year"].astype(str) + "-01-01"
    ) - pd.to_timedelta(pat["anchor_age"] * 365.25, unit="D")

    df = icu.merge(
        adm[["hadm_id", "admittime", "dischtime"]],
        on="hadm_id", how="left"
    ).merge(
        pat[["subject_id", "anchor_age", "anchor_year", "gender", "dob_proxy"]],
        on="subject_id", how="left"
    )

    df["los_hours"] = (
        (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0
    )
    df["age"] = (
        df["anchor_age"] +
        (df["intime"].dt.year - df["anchor_year"])
    ).clip(lower=0, upper=120)

    n_total = len(df)

    # Exclusions
    df = df[df["age"] >= 18].copy()
    n_excl_age = n_total - len(df)

    df = df[df["los_hours"] >= 4.0].copy()
    n_excl_los = n_total - n_excl_age - len(df)

    df = (
        df.sort_values("intime")
        .groupby("subject_id", as_index=False)
        .first()
    )
    n_excl_readmit = n_total - n_excl_age - n_excl_los - len(df)

    print(
        f"Cohort: total={n_total:,}  excl_age={n_excl_age:,}  "
        f"excl_los={n_excl_los:,}  excl_readmit={n_excl_readmit:,}  "
        f"final={len(df):,}"
    )

    result = df[[
        "stay_id", "subject_id", "hadm_id",
        "age", "gender", "intime", "outtime", "los_hours"
    ]].copy()

    if cache_path:
        result.to_parquet(cache_path, index=False)

    return result
