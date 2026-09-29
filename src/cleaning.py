"""Cleaning & validation for the production dataset.

Every rule is logged so the dashboard can show exactly what was changed and why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHIFT_MIN = 480
VALID_SHIFTS = ["Morning", "Afternoon", "Night"]
VALID_STATUS = ["Up to Date", "Due Soon", "Overdue", "In Progress"]
TEMP_RANGE = (20.0, 120.0)          # physically plausible machine body temperature


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean_df, log_df)."""
    df = raw.copy()
    log: list[dict] = []

    def note(step, rows, action):
        log.append({"Step": step, "Rows affected": int(rows), "Action": action})

    note("Raw rows loaded", len(df), "—")

    # 1. Standardise text categories
    before = df["Shift"].copy()
    df["Shift"] = df["Shift"].astype(str).str.strip().str.title()
    note("Shift labels standardised", (before != df["Shift"]).sum(),
         "Trimmed whitespace and fixed case (e.g. 'NIGHT', ' night ' → 'Night')")

    before = df["Machine_ID"].copy()
    df["Machine_ID"] = (df["Machine_ID"].astype(str).str.strip().str.upper()
                        .str.replace("-", "", regex=False))
    note("Machine IDs standardised", (before != df["Machine_ID"]).sum(),
         "Trimmed, upper-cased and removed hyphens (e.g. 'm-03' → 'M03')")

    before = df["Maintenance_Status"].copy()
    status_map = {s.lower(): s for s in VALID_STATUS}
    df["Maintenance_Status"] = df["Maintenance_Status"].astype(str).str.strip().str.lower().map(status_map)
    note("Maintenance status standardised", (before != df["Maintenance_Status"]).sum(),
         "Mapped to canonical labels")

    # 2. Parse mixed date formats
    iso = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    note("Dates in DD/MM/YYYY format", iso.isna().sum() - (iso.isna() & dmy.isna()).sum(),
         "Parsed alternate format and unified to ISO dates")
    df["Date"] = iso.fillna(dmy)
    bad_date = df["Date"].isna()
    note("Unparseable dates dropped", bad_date.sum(), "Removed")
    df = df[~bad_date]

    # 3. Numeric coercion
    for c in ["Machine_Output", "Downtime", "Temperature", "Production_Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. Exact duplicates (after standardisation, so re-uploads are caught)
    dup = df.duplicated()
    note("Exact duplicate rows removed", dup.sum(), "Double-uploaded records dropped")
    df = df[~dup]
    key_dup = df.duplicated(["Date", "Machine_ID", "Shift"], keep="first")
    note("Duplicate Date+Machine+Shift keys removed", key_dup.sum(), "Kept first record")
    df = df[~key_dup]

    # 5. Impossible values -> NaN (then imputed)
    bad_temp = ~df["Temperature"].between(*TEMP_RANGE) & df["Temperature"].notna()
    note("Temperature sensor faults", bad_temp.sum(),
         f"Values outside {TEMP_RANGE[0]:.0f}–{TEMP_RANGE[1]:.0f} °C (e.g. 999, -40, 0) set to missing")
    df.loc[bad_temp, "Temperature"] = np.nan

    bad_dt = ~df["Downtime"].between(0, SHIFT_MIN) & df["Downtime"].notna()
    note("Impossible downtime", bad_dt.sum(),
         f"Negative or > {SHIFT_MIN} min in a {SHIFT_MIN}-min shift set to missing")
    df.loc[bad_dt, "Downtime"] = np.nan

    bad_out = df["Machine_Output"] > df["Production_Volume"]
    note("Good output > total volume", bad_out.sum(),
         "Good units cannot exceed total units; Machine_Output set to missing")
    df.loc[bad_out, "Machine_Output"] = np.nan

    # 6. Missing-value treatment
    miss_op = df["Operator"].isna()
    df["Operator"] = df["Operator"].fillna("Unknown")
    note("Missing operator", miss_op.sum(),
         "Labelled 'Unknown' (kept — not guessed, to avoid attributing work to an individual)")

    miss_t = df["Temperature"].isna()
    df = df.sort_values(["Machine_ID", "Date", "Shift"])
    df["Temperature"] = (df.groupby("Machine_ID")["Temperature"]
                         .transform(lambda s: s.interpolate(limit_direction="both")))
    note("Missing temperature", miss_t.sum(), "Linearly interpolated within the same machine's time series")

    miss_dt = df["Downtime"].isna()
    # Median downtime for the same machine & maintenance status
    df["Downtime"] = df["Downtime"].fillna(
        df.groupby(["Machine_ID", "Maintenance_Status"])["Downtime"].transform("median"))
    note("Missing downtime", miss_dt.sum(), "Imputed with the machine × maintenance-status median")

    miss_out = df["Machine_Output"].isna()
    q = (df["Machine_Output"] / df["Production_Volume"])
    q_med = q.groupby([df["Machine_ID"], df["Date"].dt.to_period("M")]).transform("median")
    df["Machine_Output"] = df["Machine_Output"].fillna((df["Production_Volume"] * q_med).round())
    note("Missing good output", miss_out.sum(),
         "Imputed as volume × that machine's monthly median yield")

    df["Imputed"] = miss_t | miss_dt | miss_out | bad_temp | bad_dt | bad_out

    # 7. Final validation
    unknown_cat = (~df["Shift"].isin(VALID_SHIFTS)) | df["Maintenance_Status"].isna()
    note("Rows with unknown categories dropped", unknown_cat.sum(), "Removed")
    df = df[~unknown_cat]
    df = df.dropna(subset=["Machine_Output", "Downtime", "Temperature", "Production_Volume"])
    df["Shift"] = pd.Categorical(df["Shift"], VALID_SHIFTS, ordered=True)
    df["Machine_Output"] = df["Machine_Output"].astype(int)
    df["Production_Volume"] = df["Production_Volume"].astype(int)
    df = df.sort_values(["Date", "Machine_ID", "Shift"]).reset_index(drop=True)
    note("Clean rows", len(df), "Final analytical dataset")

    return df, pd.DataFrame(log)


def validation_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Assertions run on the cleaned data — all should pass."""
    n_days = df["Date"].nunique()
    checks = [
        ("No missing values in analytical columns", df.drop(columns="Imputed").isna().sum().sum() == 0),
        ("Downtime within 0–480 min", df["Downtime"].between(0, SHIFT_MIN).all()),
        ("Good output ≤ total volume", (df["Machine_Output"] <= df["Production_Volume"]).all()),
        ("Temperature within plausible range", df["Temperature"].between(*TEMP_RANGE).all()),
        ("Unique Date + Machine + Shift key", not df.duplicated(["Date", "Machine_ID", "Shift"]).any()),
        ("Only valid shift labels", df["Shift"].isin(VALID_SHIFTS).all()),
        ("Only valid maintenance labels", df["Maintenance_Status"].isin(VALID_STATUS).all()),
        (f"Coverage: {n_days} days × {df['Machine_ID'].nunique()} machines × 3 shifts",
         len(df) >= 0.98 * n_days * df["Machine_ID"].nunique() * 3),
    ]
    return pd.DataFrame(checks, columns=["Check", "Passed"])
