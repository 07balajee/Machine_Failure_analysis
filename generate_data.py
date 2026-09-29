"""
Synthetic data generator for DS_Day01_26 — "Manufacturing: Why is production falling?"

Produces one row per Machine x Shift x Day for calendar year 2025
(8 machines x 3 shifts x 365 days = 8,760 rows before dirt is injected).

Column semantics (documented assumptions, used consistently in the analysis):
    Date               calendar date of the shift
    Machine_ID         M01..M08
    Machine_Output     GOOD units produced in the shift (passed quality check)
    Downtime           minutes the machine was stopped during the 480-min shift
    Shift              Morning / Afternoon / Night
    Operator           anonymised operator code (OP01..OP15)
    Temperature        machine body temperature, deg C
    Maintenance_Status Up to Date | Due Soon | Overdue | In Progress
                       ("In Progress" = planned maintenance performed in that shift)
    Production_Volume  TOTAL units produced in the shift (good + rejected)

Hidden "ground truth" story planted in the data (the analysis must rediscover it):
  * Up to ~mid-June the plant is stable.
  * From 2025-06-16 the plant switches to a deferred-maintenance policy for
    budget reasons: planned maintenance on M03 and M06 (the two highest-capacity
    presses) is frozen for the rest of the year, and M05's service interval is
    stretched from ~30 to ~60 days.
  * Overdue machines run hotter, break down more often (unplanned downtime),
    and produce more rejects -> lower Availability AND lower Quality.
  * Summer ambient heat adds a few degrees to every machine (a confounder:
    temperature rises plant-wide, but only neglected machines lose output).
  * Operators are rostered to fixed shift/machine groups, so raw operator
    averages are confounded by which machines they run. Individual operator
    skill effects are deliberately tiny (+/-1.5%).
  * Night shift has a small genuine performance dip (~3%), constant all year —
    it is NOT the cause of the decline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 26
RNG = np.random.default_rng(SEED)

START, END = "2025-01-01", "2025-12-31"
SHIFT_MIN = 480
SHIFTS = ["Morning", "Afternoon", "Night"]
POLICY_CHANGE = pd.Timestamp("2025-06-16")

# Ideal run rate (units / running minute) per machine
MACHINES = {
    "M01": 1.10, "M02": 1.00, "M03": 1.45, "M04": 0.95,
    "M05": 1.20, "M06": 1.40, "M07": 1.05, "M08": 0.90,
}
# Maintenance interval (days) before / after the policy change
INTERVAL_BEFORE = 30
INTERVAL_AFTER = {"M03": None, "M06": None, "M05": 60}   # None = maintenance frozen

# Operators rostered in groups: (shift, machine group)
OPERATOR_ROSTER = {
    ("Morning", 0): ["OP01", "OP02"], ("Morning", 1): ["OP03", "OP04", "OP05"],
    ("Afternoon", 0): ["OP06", "OP07"], ("Afternoon", 1): ["OP08", "OP09", "OP10"],
    ("Night", 0): ["OP11", "OP12"], ("Night", 1): ["OP13", "OP14", "OP15"],
}
MACHINE_GROUP = {"M01": 0, "M02": 0, "M04": 0, "M07": 0,   # group 0: stable machines
                 "M03": 1, "M05": 1, "M06": 1, "M08": 1}   # group 1: includes neglected ones
OPERATOR_SKILL = {f"OP{i:02d}": s for i, s in
                  enumerate(RNG.normal(0, 0.012, 15).clip(-0.015, 0.015), start=1)}
SHIFT_PERF = {"Morning": 1.00, "Afternoon": 0.99, "Night": 0.97}


def maintenance_days(machine: str, dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Return the set of dates on which planned maintenance happens."""
    days, d = set(), dates[0] + pd.Timedelta(days=int(RNG.integers(0, INTERVAL_BEFORE)))
    while d <= dates[-1]:
        days.add(d)
        interval = INTERVAL_AFTER.get(machine, INTERVAL_BEFORE) if d >= POLICY_CHANGE else INTERVAL_BEFORE
        if interval is None:
            break
        d += pd.Timedelta(days=int(interval + RNG.integers(-3, 4)))
    return days


def status_from_age(age: int, in_progress: bool) -> str:
    """Status as the plant's CMMS would report it against the 30-day OEM interval."""
    if in_progress:
        return "In Progress"
    if age > INTERVAL_BEFORE + 5:
        return "Overdue"
    if age >= INTERVAL_BEFORE - 5:
        return "Due Soon"
    return "Up to Date"


def generate() -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="D")
    doy = dates.dayofyear.to_numpy()
    # Seasonal ambient temperature effect (peaks late July)
    ambient = 4.0 * np.sin(2 * np.pi * (doy - 110) / 365)

    rows = []
    for m, rate in MACHINES.items():
        mdays = maintenance_days(m, dates)
        base_temp = RNG.uniform(62, 68)
        last_service = dates[0] - pd.Timedelta(days=int(RNG.integers(0, 20)))
        for di, d in enumerate(dates):
            is_mday = d in mdays
            for si, shift in enumerate(SHIFTS):
                in_progress = is_mday and shift == "Morning"
                if in_progress:
                    last_service = d
                age = (d - last_service).days
                status = status_from_age(age, in_progress)

                # Wear accumulates once past the recommended 30-day interval (capped)
                wear = min(max(0.0, age - INTERVAL_BEFORE) / 30.0, 4.0)

                temp = (base_temp + ambient[di] + 3.5 * wear
                        + (1.5 if shift == "Afternoon" else 0) + RNG.normal(0, 1.8))

                # Downtime: planned maintenance + minor stops + breakdowns
                planned = RNG.normal(150, 20) if in_progress else 0.0
                minor = RNG.gamma(2.0, 7.0)
                p_break = min(0.03 + 0.12 * wear + 0.004 * max(0, temp - 72), 0.65)
                breakdown = RNG.gamma(3.0, 30.0 + 8.0 * wear) if RNG.random() < p_break else 0.0
                downtime = float(np.clip(planned + minor + breakdown, 0, SHIFT_MIN - 30))

                group = MACHINE_GROUP[m]
                op = RNG.choice(OPERATOR_ROSTER[(shift, group)])

                run_time = SHIFT_MIN - downtime
                perf = (0.92 * SHIFT_PERF[shift] * (1 + OPERATOR_SKILL[op])
                        * (1 - 0.03 * wear) * RNG.normal(1, 0.025))
                perf = float(np.clip(perf, 0.5, 1.0))
                volume = int(round(rate * run_time * perf))

                reject_rate = float(np.clip(0.02 + 0.02 * wear
                                            + 0.002 * max(0, temp - 74)
                                            + RNG.normal(0, 0.006), 0.002, 0.25))
                output = int(round(volume * (1 - reject_rate)))

                rows.append({
                    "Date": d, "Machine_ID": m, "Machine_Output": output,
                    "Downtime": round(downtime, 1), "Shift": shift, "Operator": op,
                    "Temperature": round(temp, 1), "Maintenance_Status": status,
                    "Production_Volume": volume,
                })
    return pd.DataFrame(rows)


def inject_dirt(df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic data-quality problems for the cleaning step to catch."""
    df = df.copy()
    n = len(df)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d").astype(object)

    def pick(frac):
        return RNG.choice(n, size=int(n * frac), replace=False)

    # 1) Mixed date formats (ERP export from a second system)
    idx = pick(0.03)
    df.loc[idx, "Date"] = pd.to_datetime(df.loc[idx, "Date"]).dt.strftime("%d/%m/%Y")
    # 2) Inconsistent categorical labels
    idx = pick(0.02)
    df.loc[idx, "Shift"] = df.loc[idx, "Shift"].map(
        lambda s: RNG.choice([s.lower(), s.upper(), f" {s} "]))
    idx = pick(0.015)
    df.loc[idx, "Machine_ID"] = df.loc[idx, "Machine_ID"].map(
        lambda s: RNG.choice([s.lower(), s.replace("M", "M-"), f"{s} "]))
    idx = pick(0.01)
    df.loc[idx, "Maintenance_Status"] = df.loc[idx, "Maintenance_Status"].str.lower()
    # 3) Missing values
    df = df.astype({"Temperature": object, "Downtime": object, "Machine_Output": object})
    df.loc[pick(0.025), "Temperature"] = np.nan
    df.loc[pick(0.01), "Operator"] = np.nan
    df.loc[pick(0.008), "Downtime"] = np.nan
    df.loc[pick(0.005), "Machine_Output"] = np.nan
    # 4) Sensor faults / impossible values
    df.loc[pick(0.003), "Temperature"] = RNG.choice([999.0, -40.0, 0.0], size=int(n * 0.003))
    df.loc[pick(0.002), "Downtime"] = RNG.choice([-15.0, 600.0, 1440.0], size=int(n * 0.002))
    idx = pick(0.002)                       # good units > total units (swapped fields)
    df.loc[idx, "Machine_Output"] = df.loc[idx, "Production_Volume"].astype(float) * 1.15
    # 5) Duplicate records (double upload)
    dups = df.sample(frac=0.012, random_state=SEED)
    df = pd.concat([df, dups], ignore_index=True)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    clean = generate()
    dirty = inject_dirt(clean)
    dirty.to_csv(out / "production_data.csv", index=False)
    print(f"Wrote {len(dirty):,} rows -> {out / 'production_data.csv'}")
