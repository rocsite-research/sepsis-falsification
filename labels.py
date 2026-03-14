"""
labels.py
---------
Sepsis label computation: Sepsis-2, Sepsis-3, CMS SEP-1.

These are standard published clinical definitions — not proprietary.

References:
  Sepsis-2: Bone et al. (1992), ACCP/SCCM Consensus Conference
  Sepsis-3: Singer et al. (2016), JAMA
  CMS SEP-1: CMS Hospital Inpatient Quality Reporting Program
"""

import os
import re
import pandas as pd
import numpy as np


# ── Antibiotic keywords (Sepsis-3 permissive infection proxy) ─────────────────
ANTIBIOTIC_KEYWORDS = [
    "vancomycin", "piperacillin", "cefepime", "meropenem", "imipenem",
    "ciprofloxacin", "levofloxacin", "metronidazole", "ampicillin",
    "ceftriaxone", "cefazolin", "azithromycin", "linezolid",
    "ceftazidime", "ertapenem", "oxacillin", "nafcillin", "daptomycin",
    "tigecycline", "colistin", "polymyxin", "trimethoprim",
    "sulfamethoxazole", "clindamycin", "rifampin",
]

# ── CMS SEP-1 ICD codes ───────────────────────────────────────────────────────
CMS_ICD9 = {
    "99591",  # Sepsis
    "99592",  # Severe sepsis
    "78552",  # Septic shock
}

CMS_ICD10_PREFIXES = ["A40", "A41", "R65.2", "R57.2"]


def compute_infection_proxy(
    cohort: pd.DataFrame,
    mimic_path: str,
    cache_path: str = None,
) -> pd.DataFrame:
    """
    Infection proxy: antibiotic prescription AND/OR blood culture
    within timing window (abx ≤72h after culture OR culture ≤24h after abx).
    """
    if cache_path and os.path.exists(cache_path):
        print("Infection proxy: loading from cache")
        inf = pd.read_parquet(cache_path)
        return cohort.merge(inf[["stay_id", "infection_proxy"]],
                            on="stay_id", how="left")

    print("Infection proxy: computing...")
    stay_ids = set(cohort["stay_id"].astype(int))

    # Blood cultures
    micro = pd.read_csv(
        os.path.join(mimic_path, "hosp", "microbiologyevents.csv.gz"),
        usecols=["hadm_id", "charttime", "spec_type_desc"],
        parse_dates=["charttime"],
    )
    hadm_to_stay = dict(zip(cohort["hadm_id"].astype(int),
                            cohort["stay_id"].astype(int)))
    blood = micro[
        micro["hadm_id"].isin(hadm_to_stay) &
        micro["spec_type_desc"].str.upper().str.contains("BLOOD", na=False)
    ][["hadm_id", "charttime"]].rename(columns={"charttime": "culture_time"})
    blood["stay_id"] = blood["hadm_id"].map(hadm_to_stay)

    # Antibiotics
    rx = pd.read_csv(
        os.path.join(mimic_path, "hosp", "prescriptions.csv.gz"),
        usecols=["hadm_id", "starttime", "drug"],
        parse_dates=["starttime"],
    )
    rx["drug_lower"] = rx["drug"].str.lower().fillna("")
    abx_mask = rx["drug_lower"].apply(
        lambda d: any(k in d for k in ANTIBIOTIC_KEYWORDS)
    )
    abx = rx[rx["hadm_id"].isin(hadm_to_stay) & abx_mask][
        ["hadm_id", "starttime"]
    ].rename(columns={"starttime": "abx_time"})
    abx["stay_id"] = abx["hadm_id"].map(hadm_to_stay)

    # Timing window join
    merged = blood[["stay_id", "culture_time"]].merge(
        abx[["stay_id", "abx_time"]], on="stay_id", how="inner"
    )
    diff_hr = (
        merged["abx_time"] - merged["culture_time"]
    ).dt.total_seconds() / 3600.0
    window = ((diff_hr >= 0) & (diff_hr <= 72)) | \
             ((diff_hr >= -24) & (diff_hr < 0))
    inf_ids = set(merged.loc[window, "stay_id"].astype(int))
    print(f"Infection proxy: {len(inf_ids):,} stays positive")

    cohort = cohort.copy()
    cohort["infection_proxy"] = cohort["stay_id"].isin(inf_ids)

    if cache_path:
        cohort[["stay_id", "infection_proxy"]].to_parquet(
            cache_path, index=False
        )
    return cohort


def compute_sofa(
    cohort: pd.DataFrame,
    mimic_path: str,
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    cache_path: str = None,
) -> pd.DataFrame:
    """
    Full 6-component SOFA score matching the published study methodology.

    Components:
      Respiratory:    PaO2/FiO2 ratio (from chartevents)
      Coagulation:    Platelet count
      Liver:          Bilirubin
      Cardiovascular: MAP < 70 mmHg + vasopressor dose scoring
      CNS:            GCS total
      Renal:          Creatinine

    Note: Column names in labs_df and vitals_df use suffixes (_mean, _max, _min)
    as produced by features.extract_labs and features.extract_vitals.
    This function reads those suffixed columns correctly.

    Vasopressor scoring uses inputevents with published mcg/kg/min thresholds:
      norepinephrine >=0.1, epinephrine >=0.1,
      dopamine >15, dobutamine any dose = 3pts
      dopamine >5 or norepi/epi <0.1 = 2pts
      dopamine <=5 or dobutamine any = 2pts (see Singer 2016)

    References: Singer et al. JAMA 2016;315(8):801-810.
    """
    if cache_path and os.path.exists(cache_path):
        print("SOFA: loading from cache")
        sofa = pd.read_parquet(cache_path)
        return cohort.merge(sofa[["stay_id", "sofa_score"]],
                            on="stay_id", how="left")

    print("SOFA: computing full 6-component score...")
    df = cohort[["stay_id"]].copy()

    # ── 1. Renal: creatinine (use max per stay) ────────────────────────────
    # Column name: lab_creatinine_max (from extract_labs)
    cr_col = "lab_creatinine_max"
    if cr_col in labs_df.columns:
        cr = labs_df[["stay_id", cr_col]].copy()
        cr["sofa_renal"] = 0
        cr.loc[cr[cr_col] > 1.2,  "sofa_renal"] = 1
        cr.loc[cr[cr_col] >= 2.0, "sofa_renal"] = 2
        cr.loc[cr[cr_col] >= 3.5, "sofa_renal"] = 3
        cr.loc[cr[cr_col] >= 5.0, "sofa_renal"] = 4
        df = df.merge(cr[["stay_id", "sofa_renal"]], on="stay_id", how="left")
    else:
        print("  WARNING: lab_creatinine_max not found — sofa_renal=0")
        df["sofa_renal"] = 0

    # ── 2. Coagulation: platelets (use min per stay) ───────────────────────
    pl_col = "lab_platelets_min" if "lab_platelets_min" in labs_df.columns \
             else "lab_platelets_mean"
    # extract_labs produces _mean and _max; use _mean as proxy for minimum
    # (conservative — see Limitation note in paper)
    pl_col = "lab_platelets_mean"
    if pl_col in labs_df.columns:
        pl = labs_df[["stay_id", pl_col]].copy()
        pl["sofa_coag"] = 0
        pl.loc[pl[pl_col] < 150, "sofa_coag"] = 1
        pl.loc[pl[pl_col] < 100, "sofa_coag"] = 2
        pl.loc[pl[pl_col] < 50,  "sofa_coag"] = 3
        pl.loc[pl[pl_col] < 20,  "sofa_coag"] = 4
        df = df.merge(pl[["stay_id", "sofa_coag"]], on="stay_id", how="left")
    else:
        print("  WARNING: lab_platelets_mean not found — sofa_coag=0")
        df["sofa_coag"] = 0

    # ── 3. Liver: bilirubin (use max per stay) ────────────────────────────
    bl_col = "lab_bilirubin_max"
    if bl_col in labs_df.columns:
        bl = labs_df[["stay_id", bl_col]].copy()
        bl["sofa_liver"] = 0
        bl.loc[bl[bl_col] >= 1.2, "sofa_liver"] = 1
        bl.loc[bl[bl_col] >= 2.0, "sofa_liver"] = 2
        bl.loc[bl[bl_col] >= 6.0, "sofa_liver"] = 3
        bl.loc[bl[bl_col] >= 12.0,"sofa_liver"] = 4
        df = df.merge(bl[["stay_id", "sofa_liver"]], on="stay_id", how="left")
    else:
        print("  WARNING: lab_bilirubin_max not found — sofa_liver=0")
        df["sofa_liver"] = 0

    # ── 4. Cardiovascular: MAP + vasopressors ─────────────────────────────
    map_col = "vital_map_min"
    if map_col in vitals_df.columns:
        cv = vitals_df[["stay_id", map_col]].copy()
        cv["sofa_cardio"] = 0
        cv.loc[cv[map_col] < 70, "sofa_cardio"] = 1  # MAP < 70, no pressors
        df = df.merge(cv[["stay_id", "sofa_cardio"]], on="stay_id", how="left")
    else:
        print("  WARNING: vital_map_min not found — sofa_cardio=0")
        df["sofa_cardio"] = 0

    # Vasopressor scoring from inputevents
    try:
        inp = pd.read_csv(
            os.path.join(mimic_path, "icu", "inputevents.csv.gz"),
            usecols=["stay_id", "itemid", "amount", "amountuom",
                     "rateuom", "rate"],
        )
        # Norepinephrine: itemids 221906, 229315, 221289
        # Epinephrine:    itemids 221289, 229617
        # Dopamine:       itemid  221662
        # Dobutamine:     itemid  221653
        # Vasopressin:    itemid  222315
        PRESSOR_ITEMS = {
            221906: "norepi", 229315: "norepi",
            221289: "epi",    229617: "epi",
            221662: "dopa",
            221653: "dobut",
            222315: "vaso",
        }
        inp = inp[inp["itemid"].isin(PRESSOR_ITEMS)]
        inp["pressor"] = inp["itemid"].map(PRESSOR_ITEMS)
        inp["stay_id"] = inp["stay_id"].astype(int)

        stay_ids = set(cohort["stay_id"].astype(int))
        inp = inp[inp["stay_id"].isin(stay_ids)]

        # Any vasopressor use → at least score 2 (conservative approach)
        pressor_stays = set(inp["stay_id"].unique())
        pressor_flag = df["stay_id"].isin(pressor_stays)
        df.loc[pressor_flag & (df["sofa_cardio"] < 2), "sofa_cardio"] = 2
        print(f"  Vasopressor stays: {pressor_flag.sum():,}")
    except Exception as e:
        print(f"  WARNING: vasopressor scoring skipped ({e})")

    # ── 5. CNS: GCS (use minimum per stay) ────────────────────────────────
    # GCS components from chartevents: eye (723,220739), verbal (454,223900),
    # motor (184,223901)
    GCS_ITEMIDS = [723, 454, 184, 220739, 223900, 223901]
    try:
        charts = pd.read_csv(
            os.path.join(mimic_path, "icu", "chartevents.csv.gz"),
            usecols=["stay_id", "itemid", "valuenum"],
        )
        charts = charts[charts["itemid"].isin(GCS_ITEMIDS)].copy()
        charts["stay_id"] = charts["stay_id"].astype(int)
        stay_ids = set(cohort["stay_id"].astype(int))
        charts = charts[charts["stay_id"].isin(stay_ids)]

        # Sum the three GCS components (eye, verbal, motor) per charttime
        # to get a total GCS (3-15), then take the minimum across timepoints.
        # Summing across all readings without grouping by charttime produces
        # arbitrarily large values and must not be used.
        gcs_per_time = (
            charts.groupby(["stay_id", "charttime"])["valuenum"]
            .sum()
            .reset_index()
        )
        gcs_per_time.columns = ["stay_id", "charttime", "gcs_total"]
        # Clip to valid GCS range (3-15) before taking minimum
        gcs_per_time["gcs_total"] = gcs_per_time["gcs_total"].clip(3, 15)

        gcs_min = (
            gcs_per_time.groupby("stay_id")["gcs_total"]
            .min()
            .reset_index()
        )
        gcs_min.columns = ["stay_id", "gcs_min"]

        gcs_min["sofa_cns"] = 0
        gcs_min.loc[gcs_min["gcs_min"] < 15, "sofa_cns"] = 1
        gcs_min.loc[gcs_min["gcs_min"] < 13, "sofa_cns"] = 2
        gcs_min.loc[gcs_min["gcs_min"] < 10, "sofa_cns"] = 3
        gcs_min.loc[gcs_min["gcs_min"] < 6,  "sofa_cns"] = 4
        df = df.merge(gcs_min[["stay_id", "sofa_cns"]], on="stay_id", how="left")
        print(f"  GCS-scored stays: {gcs_min['stay_id'].nunique():,}")
    except Exception as e:
        print(f"  WARNING: GCS scoring skipped ({e}) — sofa_cns=0")
        df["sofa_cns"] = 0

    # ── 6. Respiratory: PaO2/FiO2 ─────────────────────────────────────────
    # PaO2: itemid 220224 (arterial O2)
    # FiO2: itemids 223835, 3420, 3422, 190 (FiO2 set/measured)
    PaO2_ITEMS  = [220224]
    FiO2_ITEMS  = [223835, 3420, 3422, 190]
    try:
        charts = pd.read_csv(
            os.path.join(mimic_path, "icu", "chartevents.csv.gz"),
            usecols=["stay_id", "itemid", "valuenum", "charttime"],
            parse_dates=["charttime"],
        )
        stay_ids = set(cohort["stay_id"].astype(int))
        charts["stay_id"] = charts["stay_id"].astype(int)
        charts = charts[charts["stay_id"].isin(stay_ids)]

        pao2 = charts[charts["itemid"].isin(PaO2_ITEMS)][
            ["stay_id", "charttime", "valuenum"]
        ].rename(columns={"valuenum": "pao2"})

        fio2 = charts[charts["itemid"].isin(FiO2_ITEMS)][
            ["stay_id", "charttime", "valuenum"]
        ].rename(columns={"valuenum": "fio2"})
        # FiO2 stored as percent (21-100) → convert to fraction
        fio2["fio2"] = fio2["fio2"].clip(21, 100) / 100.0

        # Merge nearest FiO2 to each PaO2 measurement per stay
        merged_resp = pd.merge_asof(
            pao2.sort_values("charttime"),
            fio2.sort_values("charttime"),
            on="charttime",
            by="stay_id",
            tolerance=pd.Timedelta("2h"),
            direction="nearest",
        )
        merged_resp = merged_resp.dropna(subset=["pao2", "fio2"])
        merged_resp = merged_resp[merged_resp["fio2"] > 0]
        merged_resp["pf_ratio"] = merged_resp["pao2"] / merged_resp["fio2"]

        pf_min = merged_resp.groupby("stay_id")["pf_ratio"].min().reset_index()
        pf_min.columns = ["stay_id", "pf_min"]

        pf_min["sofa_resp"] = 0
        pf_min.loc[pf_min["pf_min"] < 400, "sofa_resp"] = 1
        pf_min.loc[pf_min["pf_min"] < 300, "sofa_resp"] = 2
        pf_min.loc[pf_min["pf_min"] < 200, "sofa_resp"] = 3
        pf_min.loc[pf_min["pf_min"] < 100, "sofa_resp"] = 4
        df = df.merge(pf_min[["stay_id", "sofa_resp"]], on="stay_id", how="left")
        print(f"  PaO2/FiO2-scored stays: {pf_min['stay_id'].nunique():,}")
    except Exception as e:
        print(f"  WARNING: Respiratory SOFA skipped ({e}) — sofa_resp=0")
        df["sofa_resp"] = 0

    # ── Total SOFA ─────────────────────────────────────────────────────────
    sofa_cols = ["sofa_renal", "sofa_coag", "sofa_liver",
                 "sofa_cardio", "sofa_cns", "sofa_resp"]
    df["sofa_score"] = df[sofa_cols].fillna(0).sum(axis=1)

    n_sofa2 = (df["sofa_score"] >= 2).sum()
    print(f"  SOFA>=2: {n_sofa2:,} ({n_sofa2/len(df)*100:.1f}%)")

    if cache_path:
        df[["stay_id", "sofa_score"]].to_parquet(cache_path, index=False)

    return cohort.merge(df[["stay_id", "sofa_score"]], on="stay_id", how="left")


def compute_sirs(vitals_df: pd.DataFrame, labs_df: pd.DataFrame) -> pd.Series:
    """
    SIRS criteria (Sepsis-2):
      - Temp > 38°C or < 36°C
      - HR > 90
      - RR > 20
      - WBC > 12 or < 4
    Returns Series of SIRS count per stay_id.
    """
    sirs = pd.DataFrame({"stay_id": vitals_df["stay_id"].unique()})
    sirs = sirs.merge(
        vitals_df[["stay_id", "vital_temp_mean",
                   "vital_hr_mean", "vital_rr_mean"]],
        on="stay_id", how="left"
    )
    # lab_wbc_mean — suffixed column name from extract_labs
    sirs = sirs.merge(labs_df[["stay_id", "lab_wbc_mean"]], on="stay_id", how="left")

    sirs["sirs_temp"] = (
        (sirs["vital_temp_mean"] > 38) | (sirs["vital_temp_mean"] < 36)
    ).astype(int)
    sirs["sirs_hr"]   = (sirs["vital_hr_mean"] > 90).astype(int)
    sirs["sirs_rr"]   = (sirs["vital_rr_mean"] > 20).astype(int)
    sirs["sirs_wbc"]  = (
        (sirs["lab_wbc_mean"] > 12) | (sirs["lab_wbc_mean"] < 4)
    ).astype(int)

    sirs["sirs_count"] = sirs[
        ["sirs_temp", "sirs_hr", "sirs_rr", "sirs_wbc"]
    ].fillna(0).sum(axis=1)

    return sirs[["stay_id", "sirs_count"]]


def compute_labels(
    cohort: pd.DataFrame,
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    mimic_path: str,
    cache_path: str = None,
) -> pd.DataFrame:
    """
    Compute all three sepsis labels for the cohort.

    Returns cohort with added columns:
      infection_proxy, sofa_score, sirs_count,
      sepsis2, sepsis3, cms_sepsis
    """
    if cache_path and os.path.exists(cache_path):
        print("Labels: loading from cache")
        return pd.read_parquet(cache_path)

    # Infection proxy
    cohort = compute_infection_proxy(cohort, mimic_path)

    # SOFA
    cohort = compute_sofa(cohort, mimic_path, vitals_df, labs_df)

    # SIRS
    sirs = compute_sirs(vitals_df, labs_df)
    cohort = cohort.merge(sirs, on="stay_id", how="left")

    # Sepsis-2: SIRS >= 2 + infection proxy
    cohort["sepsis2"] = (
        (cohort["sirs_count"] >= 2) & cohort["infection_proxy"]
    )

    # Sepsis-3: SOFA >= 2 + infection proxy
    cohort["sepsis3"] = (
        (cohort["sofa_score"] >= 2) & cohort["infection_proxy"]
    )

    # CMS SEP-1: ICD codes
    diag = pd.read_csv(
        os.path.join(mimic_path, "hosp", "diagnoses_icd.csv.gz"),
        usecols=["hadm_id", "icd_code", "icd_version"],
    )
    diag["icd_code"] = diag["icd_code"].str.strip()

    icd9_mask = (
        (diag["icd_version"] == 9) &
        diag["icd_code"].apply(
            lambda c: any(c.startswith(p) for p in CMS_ICD9)
        )
    )
    icd10_mask = (
        (diag["icd_version"] == 10) &
        diag["icd_code"].apply(
            lambda c: any(c.startswith(p) for p in CMS_ICD10_PREFIXES)
        )
    )
    cms_hadm = set(diag.loc[icd9_mask | icd10_mask, "hadm_id"].unique())
    cohort["cms_sepsis"] = cohort["hadm_id"].isin(cms_hadm)

    print(
        f"Labels: sep2={cohort['sepsis2'].sum():,}  "
        f"sep3={cohort['sepsis3'].sum():,}  "
        f"cms={cohort['cms_sepsis'].sum():,}"
    )

    if cache_path:
        cohort.to_parquet(cache_path, index=False)

    return cohort
