"""Analysis layer: feature engineering, decline detection, loss attribution,
downtime / shift / operator / maintenance analysis and OEE."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SHIFT_MIN = 480
UNPLANNED_BREAKDOWN_MIN = 60      # unplanned stop longer than this = "breakdown shift"
PLANNED_MAINT_MIN = 150           # typical planned-maintenance duration (used to split downtime)


# --------------------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Run_Time"] = SHIFT_MIN - df["Downtime"]
    df["Rejects"] = df["Production_Volume"] - df["Machine_Output"]
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["Week"] = df["Date"].dt.to_period("W-SUN").dt.start_time
    df["DayOfWeek"] = pd.Categorical(df["Date"].dt.day_name(),
                                     ["Monday", "Tuesday", "Wednesday", "Thursday",
                                      "Friday", "Saturday", "Sunday"], ordered=True)

    # Ideal rate: highest demonstrated rate (99th pct of units per running minute) per machine.
    rate = df["Production_Volume"] / df["Run_Time"].clip(lower=1)
    df["Ideal_Rate"] = rate.groupby(df["Machine_ID"]).transform(lambda s: s.quantile(0.99))

    # OEE components (planned production time = full 480-min shift)
    df["Availability"] = df["Run_Time"] / SHIFT_MIN
    df["Performance"] = (df["Production_Volume"] / (df["Ideal_Rate"] * df["Run_Time"].clip(lower=1))).clip(upper=1)
    df["Quality"] = np.where(df["Production_Volume"] > 0,
                             df["Machine_Output"] / df["Production_Volume"].replace(0, np.nan), 0)
    df["OEE"] = df["Availability"] * df["Performance"] * df["Quality"]

    # Planned vs unplanned downtime
    in_prog = df["Maintenance_Status"].eq("In Progress")
    df["Planned_Downtime"] = np.where(in_prog, np.minimum(df["Downtime"], PLANNED_MAINT_MIN), 0.0)
    df["Unplanned_Downtime"] = df["Downtime"] - df["Planned_Downtime"]
    df["Breakdown"] = (df["Unplanned_Downtime"] > UNPLANNED_BREAKDOWN_MIN).astype(int)

    # Days since last planned maintenance (derived from 'In Progress' records)
    df = df.sort_values(["Machine_ID", "Date", "Shift"])
    last = df["Date"].where(in_prog.loc[df.index]).groupby(df["Machine_ID"]).ffill()
    df["Days_Since_Maint"] = (df["Date"] - last).dt.days          # NaN before first service seen
    return df.sort_values(["Date", "Machine_ID", "Shift"]).reset_index(drop=True)


# --------------------------------------------------------------------------- decline
def daily_plant(df: pd.DataFrame) -> pd.DataFrame:
    d = df.groupby("Date").agg(Output=("Machine_Output", "sum"),
                               Downtime=("Downtime", "sum"),
                               OEE=("OEE", "mean")).reset_index()
    d["Output_7d"] = d["Output"].rolling(7, center=True, min_periods=4).mean()
    d["Output_28d"] = d["Output"].rolling(28, center=True, min_periods=14).mean()
    return d


def detect_decline(daily: pd.DataFrame, baseline_end: str = "2025-04-30") -> dict:
    """Two independent methods:
    1. Piecewise-linear 'hinge' regression: flat-ish trend then a break, breakpoint chosen by min SSE.
    2. Control-chart rule: first date the 7-day mean falls below baseline mean − 3σ(7-day)
       and stays below for 14 consecutive days.
    """
    y = daily["Output"].to_numpy(float)
    t = np.arange(len(y), dtype=float)
    best = (np.inf, None, None)
    for k in range(30, len(y) - 30):
        X = np.column_stack([np.ones_like(t), t, np.maximum(0, t - k)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((y - X @ beta) ** 2).sum()
        if sse < best[0]:
            best = (sse, k, beta)
    _, k, beta = best
    fit = beta[0] + beta[1] * t + beta[2] * np.maximum(0, t - k)

    base = daily[daily["Date"] <= baseline_end]
    mu = base["Output"].mean()
    sd7 = base["Output"].std() / np.sqrt(7)
    lcl = mu - 3 * sd7
    below = (daily["Output_7d"] < lcl).to_numpy()
    run = np.convolve(below.astype(int), np.ones(14, int), "full")[: len(below)]
    idx = np.argmax(run >= 14)
    cc_date = daily["Date"].iloc[idx - 13] if run.max() >= 14 else pd.NaT

    return {
        "hinge_date": daily["Date"].iloc[k],
        "slope_before": beta[1], "slope_after": beta[1] + beta[2],
        "fit": fit, "baseline_mean": mu, "lcl": lcl, "control_date": cc_date,
    }


def split_periods(df: pd.DataFrame, decline_date: pd.Timestamp) -> pd.Series:
    return np.where(df["Date"] < decline_date, "Before decline", "After decline")


# --------------------------------------------------------------------------- losses
def machine_losses(df: pd.DataFrame, decline_date: pd.Timestamp) -> pd.DataFrame:
    """Loss = expected good output at the machine × shift pre-decline average − actual, after decline."""
    pre = df[df["Date"] < decline_date]
    post = df[df["Date"] >= decline_date]
    base = pre.groupby(["Machine_ID", "Shift"], observed=True)["Machine_Output"].mean().rename("Baseline")
    post = post.join(base, on=["Machine_ID", "Shift"])
    post["Loss"] = post["Baseline"] - post["Machine_Output"]
    out = post.groupby("Machine_ID").agg(Units_Lost=("Loss", "sum"),
                                         Expected=("Baseline", "sum")).reset_index()
    out["Loss_%"] = 100 * out["Units_Lost"] / out["Expected"]
    out = out.sort_values("Units_Lost", ascending=False).reset_index(drop=True)
    total_pos = out["Units_Lost"].clip(lower=0).sum()
    out["Share_of_Loss_%"] = 100 * out["Units_Lost"].clip(lower=0) / total_pos
    out["Cumulative_%"] = out["Share_of_Loss_%"].cumsum()
    return out


def loss_by_component(df: pd.DataFrame, decline_date: pd.Timestamp) -> pd.DataFrame:
    """Log-decomposition of the change in mean good output per machine:
    ln(G_post/G_pre) = ln(A ratio) + ln(P ratio) + ln(Q ratio)   (ideal rate constant).
    Each component's share of the units lost is proportional to its log term."""
    df = df.assign(Period=split_periods(df, decline_date))
    g = df.groupby(["Machine_ID", "Period"])[["Availability", "Performance", "Quality", "Machine_Output"]].mean()
    rows = []
    for m in df["Machine_ID"].unique():
        pre, post = g.loc[(m, "Before decline")], g.loc[(m, "After decline")]
        logs = {c: np.log(post[c] / pre[c]) for c in ["Availability", "Performance", "Quality"]}
        total_log = sum(logs.values())
        n_post = (df["Period"].eq("After decline") & df["Machine_ID"].eq(m)).sum()
        units = (pre["Machine_Output"] - post["Machine_Output"]) * n_post
        for c, v in logs.items():
            share = v / total_log if total_log != 0 else 0
            rows.append({"Machine_ID": m, "Component": c,
                         "Units_Lost": units * share if units > 0 else 0.0,
                         "Pre": pre[c], "Post": post[c]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- downtime
def downtime_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="Machine_ID", columns="Month", values="Unplanned_Downtime", aggfunc="mean")


def downtime_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return (df.groupby(by, observed=True)
              .agg(Mean_Downtime=("Downtime", "mean"),
                   Unplanned=("Unplanned_Downtime", "mean"),
                   Planned=("Planned_Downtime", "mean"),
                   Breakdown_Rate=("Breakdown", "mean"),
                   Shifts=("Downtime", "size"))
              .reset_index())


# --------------------------------------------------------------------------- shifts & operators
def adjusted_residuals(df: pd.DataFrame, keys=("Machine_ID", "Week")) -> pd.Series:
    """% difference between a shift's output and the mean output of the same machine in the
    same week (optionally also the same shift). Removes machine, time (and shift) effects so
    groups are compared like-for-like."""
    exp = df.groupby(list(keys), observed=True)["Machine_Output"].transform("mean")
    return 100 * (df["Machine_Output"] - exp) / exp


def group_compare(df: pd.DataFrame, group: str, keys=("Machine_ID", "Week")) -> tuple[pd.DataFrame, dict]:
    d = df.assign(Adj=adjusted_residuals(df, keys))
    raw_mean = d["Machine_Output"].mean()
    tbl = (d.groupby(group, observed=True)
             .agg(n=("Adj", "size"),
                  Raw_Output=("Machine_Output", "mean"),
                  Adj_Mean=("Adj", "mean"),
                  Adj_SD=("Adj", "std"))
             .reset_index())
    tbl["Raw_vs_Avg_%"] = 100 * (tbl["Raw_Output"] / raw_mean - 1)
    tbl["CI95"] = 1.96 * tbl["Adj_SD"] / np.sqrt(tbl["n"])
    tbl["Significant"] = (tbl["Adj_Mean"].abs() > tbl["CI95"])
    groups = [g["Adj"].to_numpy() for _, g in d.groupby(group, observed=True) if len(g) > 30]
    h, p = stats.kruskal(*groups)
    # epsilon-squared effect size for Kruskal-Wallis
    eps2 = (h - len(groups) + 1) / (len(d) - len(groups))
    return tbl, {"H": h, "p": p, "eps2": max(eps2, 0.0)}


def operator_mix(df: pd.DataFrame, focus=("M03", "M05", "M06")) -> pd.DataFrame:
    d = df[df["Operator"] != "Unknown"]
    mix = (d.assign(OnFocus=d["Machine_ID"].isin(focus))
             .groupby("Operator")
             .agg(Main_Shift=("Shift", lambda s: s.mode().iat[0]),
                  Shifts_Worked=("Shift", "size"),
                  Share_on_Neglected_Machines=("OnFocus", "mean"))
             .reset_index())
    mix["Share_on_Neglected_Machines"] *= 100
    return mix


# --------------------------------------------------------------------------- maintenance
def maintenance_vs_downtime(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["Days_Since_Maint"])
    d = d[d["Maintenance_Status"] != "In Progress"]
    bins = [0, 15, 30, 45, 60, 90, 120, 150, 200]
    d = d.assign(Age_Bin=pd.cut(d["Days_Since_Maint"], bins, right=False,
                                labels=[f"{a}–{b-1}" for a, b in zip(bins[:-1], bins[1:])]))
    by_age = (d.groupby("Age_Bin", observed=True)
                .agg(Unplanned=("Unplanned_Downtime", "mean"),
                     Breakdown_Rate=("Breakdown", "mean"),
                     Temperature=("Temperature", "mean"),
                     Quality=("Quality", "mean"),
                     n=("Breakdown", "size"))
                .reset_index())
    rho, p = stats.spearmanr(d["Days_Since_Maint"], d["Unplanned_Downtime"])

    by_status = downtime_summary(df, "Maintenance_Status")

    # Seasonal confound check: overdue vs up-to-date within the same months
    summer = d["Date"].dt.month.isin([6, 7, 8, 9])
    conf = (d.assign(Season=np.where(summer, "Jun–Sep (hot)", "Other months"))
              .query("Maintenance_Status in ['Up to Date', 'Overdue']")
              .groupby(["Season", "Maintenance_Status"])["Unplanned_Downtime"].mean()
              .unstack())

    # Overdue share over time by machine
    overdue = (df.assign(Overdue=df["Maintenance_Status"].eq("Overdue"))
                 .groupby(["Month", "Machine_ID"])["Overdue"].mean().mul(100).reset_index())
    mw = stats.mannwhitneyu(d.loc[d["Maintenance_Status"] == "Overdue", "Unplanned_Downtime"],
                            d.loc[d["Maintenance_Status"] == "Up to Date", "Unplanned_Downtime"],
                            alternative="greater")
    return {"by_age": by_age, "rho": rho, "p": p, "by_status": by_status,
            "season_check": conf, "overdue_share": overdue, "mw_p": mw.pvalue, "data": d}


# --------------------------------------------------------------------------- OEE
def oee_table(df: pd.DataFrame, by) -> pd.DataFrame:
    return (df.groupby(by, observed=True)[["Availability", "Performance", "Quality", "OEE"]]
              .mean().mul(100).round(1).reset_index())


# --------------------------------------------------------------------------- insights
def build_insights(df, decline, losses, comp, shift_stats, op_stats, maint, oee_pp) -> list[dict]:
    d0 = decline["hinge_date"]
    pre = df[df["Date"] < d0]
    post = df[df["Date"] >= d0]
    pre_daily = pre.groupby("Date")["Machine_Output"].sum().mean()
    post_daily = post.groupby("Date")["Machine_Output"].sum().mean()
    last_q = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=60)].groupby("Date")["Machine_Output"].sum().mean()

    top = losses.head(3)
    top_share = top["Share_of_Loss_%"].sum()
    comp_tot = comp.groupby("Component")["Units_Lost"].sum()
    comp_share = 100 * comp_tot / comp_tot.sum()

    by_age = maint["by_age"]
    fresh = by_age.iloc[0]
    old = by_age.iloc[-1]

    pp = oee_pp.set_index("Period")
    shift_tbl, shift_test = shift_stats
    op_tbl, op_test = op_stats
    sc = maint["season_check"]
    adj = df.assign(Adj=adjusted_residuals(df), Pre=df["Date"] < d0)
    night_pre = adj.loc[adj["Pre"] & adj["Shift"].eq("Night"), "Adj"].mean()
    night_post = adj.loc[~adj["Pre"] & adj["Shift"].eq("Night"), "Adj"].mean()

    return [
        {"title": "When the decline started",
         "text": f"Plant output was flat until about **{d0:%d %b %Y}**; after that it fell by about "
                 f"**{abs(decline['slope_after']):.1f} units/day each day**. The control-chart rule separately flags "
                 f"a sustained drop from **{decline['control_date']:%d %b %Y}**. Average daily output went from "
                 f"**{pre_daily:,.0f}** before the drop to **{last_q:,.0f}** in the last 60 days "
                 f"(**{100 * (last_q / pre_daily - 1):+.1f}%**)."},
        {"title": "Three machines account for most of the lost output",
         "text": f"**{', '.join(top['Machine_ID'])}** account for **{top_share:.0f}%** of the good units lost after "
                 f"the decline started (**{losses['Units_Lost'].clip(lower=0).sum():,.0f}** units in total). "
                 f"{top['Machine_ID'].iat[0]} alone ran **{top['Loss_%'].iat[0]:.1f}%** below its own baseline. "
                 "The other machines stayed close to their baselines."},
        {"title": "The loss is mostly downtime",
         "text": f"Splitting the loss into OEE components: **Availability {comp_share.get('Availability', 0):.0f}%**, "
                 f"Quality {comp_share.get('Quality', 0):.0f}%, Performance {comp_share.get('Performance', 0):.0f}%. "
                 f"Plant OEE fell from **{pp.loc['Before decline', 'OEE']:.1f}%** to **{pp.loc['After decline', 'OEE']:.1f}%**."},
        {"title": "Downtime rises steeply once a machine passes its 30-day service interval",
         "text": f"Machines serviced in the last {fresh['Age_Bin']} days lose **{fresh['Unplanned']:.0f} min** a shift to "
                 f"unplanned stops, with a breakdown in {100 * fresh['Breakdown_Rate']:.0f}% of shifts. At "
                 f"{old['Age_Bin']} days since service this becomes **{old['Unplanned']:.0f} min** and "
                 f"**{100 * old['Breakdown_Rate']:.0f}%**. The difference between overdue and up-to-date "
                 f"machines is statistically significant (Mann-Whitney p = {maint['mw_p']:.1e}). Neglected machines also run "
                 f"hotter ({fresh['Temperature']:.1f} → {old['Temperature']:.1f} °C) and scrap more parts "
                 f"(yield {100 * fresh['Quality']:.1f}% → {100 * old['Quality']:.1f}%)."},
        {"title": "Summer heat is not the explanation",
         "text": f"All machines ran hotter in summer, but only the ones overdue for service lost output. "
                 f"Within the same season, overdue machines lost **{sc.loc['Jun–Sep (hot)', 'Overdue']:.0f} vs "
                 f"{sc.loc['Jun–Sep (hot)', 'Up to Date']:.0f} min** a shift in Jun–Sep and **{sc.loc['Other months', 'Overdue']:.0f} vs "
                 f"{sc.loc['Other months', 'Up to Date']:.0f} min** in the other months. Downtime got worse into the cooler "
                 f"Oct–Dec months, which points to maintenance age as the driver, not ambient heat."},
        {"title": "Shifts and operators are not the cause",
         "text": f"Night shift produces **{night_pre:+.1f}%** compared with the same machine in the same week before the decline, "
                 f"and **{night_post:+.1f}%** after it. The gap is constant and did not drive the decline. Raw operator "
                 f"averages differ by up to **±{op_tbl['Raw_vs_Avg_%'].abs().max():.0f}%**, but only because operators are rostered "
                 f"to different machine groups. Compared with the same machine, week and shift, every operator is within "
                 f"**±{op_tbl.loc[op_tbl['Operator'] != 'Unknown', 'Adj_Mean'].abs().max():.1f}%**. "
                 f"The data does not support blaming any individual."},
    ]
