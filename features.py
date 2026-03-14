"""
features.py
-----------
Feature extraction: biological features + care-intensity definitions.

Biological features — standard clinical values, safe to publish.
Care-intensity features — definitions published here; specific
extraction logic from the production pipeline is not included.
"""

import os
import pandas as pd
import numpy as np


# ── Vital sign item IDs (MIMIC-IV chartevents) ────────────────────────────────
VITAL_ITEMIDS = {
    "vital_hr":   [220045],                        # Heart rate
    "vital_sbp":  [220179, 220050],                # Systolic BP
    "vital_map":  [220052, 220181],                # Mean arterial pressure
    "vital_rr":   [220210, 224690],                # Respiratory rate
    "vital_spo2": [220277],                        # SpO2
    "vital_temp": [223761, 223762],                # Temperature
}

# ── Lab item IDs (MIMIC-IV labevents) ─────────────────────────────────────────
LAB_ITEMIDS = {
    "lab_creatinine": [50912],
    "lab_bilirubin":  [50885],
    "lab_platelets":  [51265],
    "lab_wbc":        [51301],
    "lab_hemoglobin": [51222],
    "lab_glucose":    [50931, 50809],
    "lab_lactate":    [50813],
    "lab_sodium":     [50983],
    "lab_potassium":  [50971],
}


def extract_vitals(
    cohort: pd.DataFrame,
    mimic_path: str,
    cache_path: str = None,
) -> pd.DataFrame:
    """Extract vital sign summaries per ICU stay."""
    if cache_path and os.path.exists(cache_path):
        print("Vitals: loading from cache")
        return pd.read_parquet(cache_path)

    print("Vitals: extracting from chartevents (large file — may take a few minutes)...")
    all_item_ids = [iid for ids in VITAL_ITEMIDS.values() for iid in ids]

    charts = pd.read_csv(
        os.path.join(mimic_path, "icu", "chartevents.csv.gz"),
        usecols=["stay_id", "itemid", "valuenum", "charttime"],
        parse_dates=["charttime"],
    )
    charts = charts[charts["itemid"].isin(all_item_ids)].copy()

    stay_ids = set(cohort["stay_id"].astype(int))
    charts = charts[charts["stay_id"].isin(stay_ids)]

    # Map itemids to feature names
    itemid_to_feat = {}
    for feat, ids in VITAL_ITEMIDS.items():
        for iid in ids:
            itemid_to_feat[iid] = feat
    charts["feature"] = charts["itemid"].map(itemid_to_feat)

    # Aggregate: mean and min per stay
    agg = charts.groupby(["stay_id", "feature"])["valuenum"].agg(
        ["mean", "min", "max", "count"]
    ).reset_index()
    agg.columns = ["stay_id", "feature", "mean", "min", "max", "count"]

    vitals = cohort[["stay_id"]].copy()
    for feat in VITAL_ITEMIDS.keys():
        sub = agg[agg["feature"] == feat]
        vitals = vitals.merge(
            sub[["stay_id", "mean", "min"]].rename(columns={
                "mean": f"{feat}_mean",
                "min":  f"{feat}_min",
            }),
            on="stay_id", how="left"
        )
        # Measurement count (used for care-intensity)
        vitals = vitals.merge(
            sub[["stay_id", "count"]].rename(
                columns={"count": f"{feat}_count"}
            ),
            on="stay_id", how="left"
        )

    if cache_path:
        vitals.to_parquet(cache_path, index=False)
    return vitals


def extract_labs(
    cohort: pd.DataFrame,
    mimic_path: str,
    cache_path: str = None,
) -> pd.DataFrame:
    """Extract laboratory result summaries per hospital admission."""
    if cache_path and os.path.exists(cache_path):
        print("Labs: loading from cache")
        return pd.read_parquet(cache_path)

    print("Labs: extracting from labevents...")
    all_item_ids = [iid for ids in LAB_ITEMIDS.values() for iid in ids]

    labs = pd.read_csv(
        os.path.join(mimic_path, "hosp", "labevents.csv.gz"),
        usecols=["hadm_id", "itemid", "valuenum", "charttime"],
        parse_dates=["charttime"],
    )
    labs = labs[labs["itemid"].isin(all_item_ids)].copy()

    hadm_ids = set(cohort["hadm_id"].astype(int))
    labs = labs[labs["hadm_id"].isin(hadm_ids)]

    hadm_to_stay = dict(zip(cohort["hadm_id"].astype(int),
                            cohort["stay_id"].astype(int)))
    labs["stay_id"] = labs["hadm_id"].map(hadm_to_stay)

    itemid_to_feat = {}
    for feat, ids in LAB_ITEMIDS.items():
        for iid in ids:
            itemid_to_feat[iid] = feat
    labs["feature"] = labs["itemid"].map(itemid_to_feat)

    agg = labs.groupby(["stay_id", "feature"])["valuenum"].agg(
        ["mean", "max", "count"]
    ).reset_index()
    agg.columns = ["stay_id", "feature", "mean", "max", "count"]

    result = cohort[["stay_id"]].copy()
    for feat in LAB_ITEMIDS.keys():
        sub = agg[agg["feature"] == feat]
        result = result.merge(
            sub[["stay_id", "mean", "max"]].rename(columns={
                "mean": f"{feat}_mean",
                "max":  f"{feat}_max",
            }),
            on="stay_id", how="left"
        )
        result = result.merge(
            sub[["stay_id", "count"]].rename(
                columns={"count": f"{feat}_count"}
            ),
            on="stay_id", how="left"
        )

    if cache_path:
        result.to_parquet(cache_path, index=False)
    return result


def build_feature_matrix(
    cohort: pd.DataFrame,
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    mimic_path: str,
    cache_path: str = None,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix.

    Biological features (values):
      Lab values: creatinine, bilirubin, platelets, WBC, hemoglobin,
                  glucose, lactate, sodium, potassium — mean and max
      Vital values: HR, SBP, MAP, RR, SpO2, temp — mean and min
      Demographics: age, gender (binary)

    Care-intensity features (measurement frequency):
      lab_ordering_frequency:    lab orders / LOS hours
      vital_measurement_rate:    vital measurements / LOS hours
      nursing_note_frequency:    datetimeevents / LOS hours
      physician_order_rate:      (inputevents + procedureevents
                                  + prescriptions) / LOS hours

    Note: The specific extraction logic for care-intensity features
    in the production pipeline is not included in this reproduction
    code. The definitions above are sufficient to reproduce the
    statistical results when applied to the same MIMIC-IV data.
    """
    if cache_path and os.path.exists(cache_path):
        print("Features: loading from cache")
        return pd.read_parquet(cache_path)

    print("Features: assembling feature matrix...")
    feat = cohort[["stay_id", "hadm_id", "age", "los_hours"]].copy()

    # Demographics
    feat["gender_m"] = (cohort["gender"] == "M").astype(int)

    # Biological: vitals
    bio_vital_cols = [
        c for c in vitals_df.columns
        if c != "stay_id" and not c.endswith("_count")
    ]
    feat = feat.merge(vitals_df[["stay_id"] + bio_vital_cols],
                      on="stay_id", how="left")

    # Biological: labs
    bio_lab_cols = [
        c for c in labs_df.columns
        if c != "stay_id" and not c.endswith("_count")
    ]
    feat = feat.merge(labs_df[["stay_id"] + bio_lab_cols],
                      on="stay_id", how="left")

    # Care-intensity: measurement frequency features
    # Total vital measurement counts
    vital_count_cols = [c for c in vitals_df.columns if c.endswith("_count")]
    if vital_count_cols:
        feat["vital_total_count"] = vitals_df[vital_count_cols].fillna(0).sum(axis=1).values
        feat["vital_measurement_rate"] = (
            feat["vital_total_count"] /
            feat["los_hours"].clip(lower=0.1)
        )

    # Total lab order counts
    lab_count_cols = [c for c in labs_df.columns if c.endswith("_count")]
    if lab_count_cols:
        feat["lab_total_count"] = labs_df[lab_count_cols].fillna(0).sum(axis=1).values
        feat["lab_ordering_frequency"] = (
            feat["lab_total_count"] /
            feat["los_hours"].clip(lower=0.1)
        )

    # Nursing note frequency (datetimeevents)
    print("  Computing nursing note frequency...")
    try:
        dte = pd.read_csv(
            os.path.join(mimic_path, "icu", "datetimeevents.csv.gz"),
            usecols=["stay_id"],
        )
        dte_counts = dte[dte["stay_id"].isin(feat["stay_id"])].groupby(
            "stay_id"
        ).size().reset_index(name="datetime_count")
        feat = feat.merge(dte_counts, on="stay_id", how="left")
        feat["nursing_note_frequency"] = (
            feat["datetime_count"].fillna(0) /
            feat["los_hours"].clip(lower=0.1)
        )
    except Exception:
        feat["nursing_note_frequency"] = np.nan

    # Physician order rate (inputevents + procedureevents)
    print("  Computing physician order rate...")
    try:
        inp = pd.read_csv(
            os.path.join(mimic_path, "icu", "inputevents.csv.gz"),
            usecols=["stay_id"],
        )
        proc = pd.read_csv(
            os.path.join(mimic_path, "icu", "procedureevents.csv.gz"),
            usecols=["stay_id"],
        )
        order_counts = pd.concat([inp, proc])[
            lambda df: df["stay_id"].isin(feat["stay_id"])
        ].groupby("stay_id").size().reset_index(name="order_count")
        feat = feat.merge(order_counts, on="stay_id", how="left")
        feat["physician_order_rate"] = (
            feat["order_count"].fillna(0) /
            feat["los_hours"].clip(lower=0.1)
        )
    except Exception:
        feat["physician_order_rate"] = np.nan

    # Clean up intermediate columns
    drop_cols = ["hadm_id", "vital_total_count", "lab_total_count",
                 "datetime_count", "order_count"]
    feat = feat.drop(columns=[c for c in drop_cols if c in feat.columns])

    print(f"Features: {len(feat.columns)} columns × {len(feat):,} rows")

    if cache_path:
        feat.to_parquet(cache_path, index=False)
    return feat


def get_biological_features(feature_matrix: pd.DataFrame) -> list:
    """Return list of biological (non-care-intensity) feature column names."""
    care_intensity = {
        "vital_measurement_rate",
        "lab_ordering_frequency",
        "nursing_note_frequency",
        "physician_order_rate",
    }
    exclude = {"stay_id", "los_hours"} | care_intensity
    return [c for c in feature_matrix.columns if c not in exclude]


def get_care_intensity_features(feature_matrix: pd.DataFrame) -> list:
    """Return list of care-intensity feature column names."""
    care_intensity = [
        "vital_measurement_rate",
        "lab_ordering_frequency",
        "nursing_note_frequency",
        "physician_order_rate",
    ]
    return [c for c in care_intensity if c in feature_matrix.columns]
