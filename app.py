"""Streamlit dashboard — DS_Day01_26: Manufacturing, why is production falling?

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analysis as A
from src import model as M
from src.cleaning import clean, validation_checks

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "production_data.csv"

st.set_page_config(page_title="Why Is Production Falling?", page_icon="🏭", layout="wide")

# ----------------------------------------------------------------------------- palette
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GRAY = "#a8a69f"
STATUS = {"Up to Date": "#0ca30c", "Due Soon": "#fab219", "Overdue": "#d03b3b", "In Progress": GRAY}
PERIOD = {"Before decline": GRAY, "After decline": BLUE}
SHIFT_C = {"Morning": BLUE, "Afternoon": ORANGE, "Night": AQUA}
OEE_C = {"Availability": BLUE, "Performance": ORANGE, "Quality": AQUA}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(fig: go.Figure, height=380, legend=True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        showlegend=legend, hoverlabel=dict(font_size=12), bargap=0.25,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5, zeroline=False)
    return fig


def show(fig, **kw):
    st.plotly_chart(style(fig, **kw), width="stretch", config={"displaylogo": False})


# ----------------------------------------------------------------------------- data pipeline (cached)
@st.cache_data(show_spinner="Cleaning and analysing data…")
def load():
    raw = pd.read_csv(DATA)
    df, log = clean(raw)
    checks = validation_checks(df)
    df = A.add_features(df)
    daily = A.daily_plant(df)
    decline = A.detect_decline(daily)
    d0 = decline["hinge_date"]
    df["Period"] = A.split_periods(df, d0)
    return raw, df, log, checks, daily, decline


@st.cache_resource(show_spinner="Training Random Forest (≈20 s, first load only)…")
def train(df: pd.DataFrame):
    return M.train_and_evaluate(df)


if not DATA.exists():
    st.error("Dataset not found. Run `python generate_data.py` first.")
    st.stop()

raw, df, log, checks, daily, decline = load()
D0 = decline["hinge_date"]
losses = A.machine_losses(df, D0)
comp = A.loss_by_component(df, D0)
maint = A.maintenance_vs_downtime(df)
shift_stats = A.group_compare(df, "Shift")
op_stats = A.group_compare(df, "Operator", ("Machine_ID", "Week", "Shift"))
oee_pp = A.oee_table(df, "Period")
insights = A.build_insights(df, decline, losses, comp, shift_stats, op_stats, maint, oee_pp)
TOP = losses.head(3)["Machine_ID"].tolist()


def machine_color(m: str) -> str:
    focus = {TOP[0]: BLUE, TOP[1]: ORANGE, TOP[2]: AQUA} if len(TOP) >= 3 else {}
    return focus.get(m, GRAY)


# ----------------------------------------------------------------------------- sidebar
PAGES = ["Overview & Key Insights", "1 · Data Cleaning & Validation", "2 · Exploratory Analysis",
         "3 · When Did the Decline Start?", "4 · Machine Losses", "5 · Downtime Patterns",
         "6 · Shifts & Operators", "7 · Maintenance vs Downtime", "8 · OEE Components",
         "9 · Random Forest Model", "Action Plan"]
with st.sidebar:
    st.markdown("### 🏭 Production Decline Analysis")
    st.caption("DS_Day01_26 · Automotive Manufacturing")
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption(f"**Dataset:** synthetic, {df['Date'].min():%d %b %Y} – {df['Date'].max():%d %b %Y}  \n"
               f"{df['Machine_ID'].nunique()} machines · 3 shifts · {df['Operator'].nunique() - 1} operators  \n"
               f"{len(raw):,} raw rows → {len(df):,} clean rows")


# ============================================================================= OVERVIEW
def page_overview():
    st.title("Why is production falling?")
    st.markdown("Analysis of daily machine output, downtime, shift, operator, temperature and maintenance records "
                "to find **when** the decline started, **where** the output was lost, and **why**.")

    pre = df[df["Period"] == "Before decline"].groupby("Date")["Machine_Output"].sum().mean()
    last60 = df[df["Date"] > df["Date"].max() - pd.Timedelta(days=60)].groupby("Date")["Machine_Output"].sum().mean()
    pp = oee_pp.set_index("Period")
    c = st.columns(4)
    c[0].metric("Decline start (breakpoint)", f"{D0:%d %b %Y}",
                f"control chart: {decline['control_date']:%d %b}", delta_color="off")
    c[1].metric("Daily good output, last 60 days", f"{last60:,.0f}", f"{100 * (last60 / pre - 1):+.1f}% vs baseline")
    c[2].metric("Plant OEE after decline", f"{pp.loc['After decline', 'OEE']:.1f}%",
                f"{pp.loc['After decline', 'OEE'] - pp.loc['Before decline', 'OEE']:+.1f} pts")
    c[3].metric("Good units lost since decline", f"{losses['Units_Lost'].clip(lower=0).sum():,.0f}",
                f"{losses.head(3)['Share_of_Loss_%'].sum():.0f}% from {', '.join(TOP)}", delta_color="off")

    st.subheader("Key insights")
    for i, ins in enumerate(insights, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {ins['title']}**  \n{ins['text']}")

    st.subheader("Four key visualisations")
    l, r = st.columns(2)
    with l:
        st.markdown("**① Daily plant output: the decline begins in late July**")
        fig_decline(height=340)
    with r:
        st.markdown("**② Pareto of lost good units by machine**")
        fig_pareto(height=340)
    l, r = st.columns(2)
    with l:
        st.markdown("**③ Unplanned downtime by days since last maintenance**")
        fig_maint_age(height=340)
    with r:
        st.markdown("**④ OEE components before and after the decline**")
        fig_oee_pp(height=340)


# ----------------------------------------------------------------------------- shared figures
def fig_decline(height=420):
    fig = go.Figure()
    fig.add_scatter(x=daily["Date"], y=daily["Output"], mode="lines", name="Daily output",
                    line=dict(color=GRAY, width=1), opacity=0.6)
    fig.add_scatter(x=daily["Date"], y=daily["Output_7d"], mode="lines", name="7-day average",
                    line=dict(color=BLUE, width=2))
    fig.add_scatter(x=daily["Date"], y=decline["fit"], mode="lines", name="Piecewise-linear fit",
                    line=dict(color=ORANGE, width=2, dash="dash"))
    fig.add_hline(y=decline["lcl"], line=dict(color=RED, width=1, dash="dot"),
                  annotation_text="Lower control limit (baseline − 3σ)", annotation_position="bottom left")
    fig.add_vline(x=D0, line=dict(color="#52514e", width=1))
    fig.add_annotation(x=D0, y=1, yref="paper", text=f"Breakpoint {D0:%d %b}", showarrow=False,
                       xanchor="left", xshift=4, yanchor="top")
    fig.update_yaxes(title="Good units / day")
    fig.update_layout(hovermode="x unified")
    show(fig, height=height)


def fig_pareto(height=420):
    L = losses[losses["Units_Lost"] > 0]
    fig = go.Figure()
    fig.add_bar(x=L["Machine_ID"], y=L["Units_Lost"], name="Units lost",
                marker_color=[machine_color(m) for m in L["Machine_ID"]],
                text=[f"{v:,.0f}" for v in L["Units_Lost"]], textposition="outside",
                hovertemplate="%{x}<br>%{y:,.0f} units lost<extra></extra>")
    fig.update_yaxes(title="Good units lost since decline")
    fig.update_xaxes(title=None)
    show(fig, height=height, legend=False)
    st.caption(f"Cumulative share: " + " → ".join(f"{m} {c:.0f}%" for m, c in
                                                  zip(L["Machine_ID"], L["Cumulative_%"])))


def fig_maint_age(height=420):
    b = maint["by_age"]
    fig = go.Figure()
    fig.add_bar(x=b["Age_Bin"].astype(str), y=b["Unplanned"], name="Unplanned downtime",
                marker_color=[BLUE if i < 3 else RED for i in range(len(b))],
                customdata=np.stack([100 * b["Breakdown_Rate"], b["n"]], axis=1),
                hovertemplate="%{x} days<br>%{y:.0f} min/shift unplanned<br>"
                              "breakdown rate %{customdata[0]:.0f}%<br>n = %{customdata[1]}<extra></extra>",
                text=[f"{v:.0f}" for v in b["Unplanned"]], textposition="outside")
    fig.add_vline(x=1.5, line=dict(color="#52514e", width=1, dash="dot"))
    fig.add_annotation(x=1.5, y=1, yref="paper", text="OEM interval (30 days)", showarrow=False,
                       xanchor="left", xshift=4, yanchor="top")
    fig.update_xaxes(title="Days since last planned maintenance", type="category")
    fig.update_yaxes(title="Mean unplanned downtime (min / shift)")
    show(fig, height=height, legend=False)


def fig_oee_pp(height=420):
    t = oee_pp.melt(id_vars="Period", var_name="Metric", value_name="Value")
    t["Period"] = pd.Categorical(t["Period"], ["Before decline", "After decline"], ordered=True)
    t = t.sort_values("Period")
    fig = px.bar(t, x="Metric", y="Value", color="Period", barmode="group", text="Value",
                 color_discrete_map=PERIOD, category_orders={"Period": ["Before decline", "After decline"]})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      hovertemplate="%{x}: %{y:.1f}%<extra>%{fullData.name}</extra>")
    fig.update_yaxes(title="%", range=[70, 102])
    fig.update_xaxes(title=None)
    show(fig, height=height)


# ============================================================================= 1. CLEANING
def page_cleaning():
    st.title("1 · Data cleaning & validation")
    st.markdown("The raw export has the problems typical of a plant data system: mixed date formats, inconsistent "
                "labels, sensor faults, impossible values, missing fields and double-uploaded rows. Every rule "
                "below is applied in [src/cleaning.py](src/cleaning.py) and logged.")
    c = st.columns(4)
    c[0].metric("Raw rows", f"{len(raw):,}")
    c[1].metric("Clean rows", f"{len(df):,}")
    c[2].metric("Rows with an imputed value", f"{df['Imputed'].sum():,}", f"{100 * df['Imputed'].mean():.1f}%",
                delta_color="off")
    c[3].metric("Validation checks passed", f"{checks['Passed'].sum()}/{len(checks)}")

    st.subheader("Cleaning log")
    st.dataframe(log, hide_index=True, width="stretch")

    l, r = st.columns([1, 1])
    with l:
        st.subheader("Missing values in raw data")
        miss = raw.isna().sum().rename("Missing").to_frame()
        miss["% of rows"] = (100 * miss["Missing"] / len(raw)).round(2)
        st.dataframe(miss, width="stretch")
    with r:
        st.subheader("Post-clean validation")
        st.dataframe(checks.assign(Passed=checks["Passed"].map({True: "✅ Pass", False: "❌ Fail"})),
                     hide_index=True, width="stretch")

    st.subheader("Examples of dirty raw records")
    bad = raw[raw["Shift"].str.strip().str.title().ne(raw["Shift"])
              | raw["Date"].astype(str).str.contains("/")
              | pd.to_numeric(raw["Temperature"], errors="coerce").isin([999, -40, 0])
              | (pd.to_numeric(raw["Downtime"], errors="coerce") < 0)].head(12)
    st.dataframe(bad, hide_index=True, width="stretch")

    with st.expander("Assumptions used for the column definitions"):
        st.markdown("""
- **Machine_Output** = *good* units in the shift; **Production_Volume** = *total* units (good + rejects).
- **Downtime** = minutes stopped in a **480-minute** shift. Shifts marked `In Progress` include planned maintenance
  (about 150 min), which is separated from *unplanned* downtime.
- **Temperature** outside 20–120 °C is treated as a sensor fault.
- **Operator** missing → `Unknown`. The value is never inferred, so no work is attributed to the wrong person.
- **Days since maintenance** is derived from the most recent `In Progress` record for that machine.
""")


# ============================================================================= 2. EDA
def page_eda():
    st.title("2 · Exploratory data analysis")
    st.subheader("Descriptive statistics")
    num = ["Machine_Output", "Production_Volume", "Downtime", "Temperature",
           "Availability", "Performance", "Quality", "OEE"]
    desc = df[num].describe().T
    desc["skew"] = df[num].skew()
    st.dataframe(desc.style.format("{:,.3f}"), width="stretch")

    st.subheader("Per-machine summary")
    per = (df.groupby("Machine_ID")
             .agg(Mean_Output=("Machine_Output", "mean"), SD_Output=("Machine_Output", "std"),
                  Mean_Downtime=("Downtime", "mean"), Median_Downtime=("Downtime", "median"),
                  Mean_Temp=("Temperature", "mean"), Yield_pct=("Quality", "mean"),
                  Overdue_pct=("Maintenance_Status", lambda s: 100 * s.eq("Overdue").mean()))
             .round(2))
    per["Yield_pct"] *= 100
    st.dataframe(per, width="stretch")

    st.subheader("Distributions")
    var = st.selectbox("Variable", num, index=0)
    split = st.radio("Split by", ["Period", "Shift", "Maintenance_Status"], horizontal=True)
    cmap = {"Period": PERIOD, "Shift": SHIFT_C, "Maintenance_Status": STATUS}[split]
    l, r = st.columns(2)
    with l:
        fig = px.histogram(df, x=var, color=split, nbins=60, barmode="overlay", opacity=0.65,
                           color_discrete_map=cmap)
        fig.update_yaxes(title="Shifts")
        show(fig)
    with r:
        fig = px.box(df, x=split, y=var, color=split, color_discrete_map=cmap, points=False)
        show(fig, legend=False)

    st.subheader("Correlation matrix (Spearman)")
    cols = ["Machine_Output", "Downtime", "Unplanned_Downtime", "Temperature", "Days_Since_Maint",
            "Availability", "Performance", "Quality"]
    corr = df[cols].corr(method="spearman").round(2)
    fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale=[[0, "#0d366b"], [0.25, "#3987e5"], [0.5, "#f0efec"],
                                            [0.75, "#ec835a"], [1, "#a8281f"]])
    show(fig, height=460)
    st.caption("Machine output depends mostly on the machine's rated speed, so the pooled correlations are weaker "
               "than the within-machine effects shown in the maintenance section.")


# ============================================================================= 3. DECLINE
def page_decline():
    st.title("3 · When did the decline start?")
    st.markdown(f"""
Two independent methods were applied to the plant's **total daily good output**:

| Method | Result |
|---|---|
| Piecewise-linear (hinge) regression: breakpoint that minimises squared error | **{D0:%d %b %Y}** |
| Control chart: first date the 7-day mean falls below baseline − 3σ and stays there for 14 days | **{decline['control_date']:%d %b %Y}** |

Before the breakpoint the trend is essentially flat (**{decline['slope_before']:+.2f} units/day²**). After it,
output falls by **{abs(decline['slope_after']):.1f} good units per day, every day**. The control chart flags the
drop later because it needs the decline to become large enough to stand out from normal variation.
""")
    fig_decline()

    st.subheader("Which machines drove the decline? Weekly output indexed to each machine's Jan–May baseline")
    w = df.groupby(["Week", "Machine_ID"])["Machine_Output"].mean().reset_index()
    base = df[df["Date"] < "2025-06-01"].groupby("Machine_ID")["Machine_Output"].mean()
    w["Index"] = 100 * w["Machine_Output"] / w["Machine_ID"].map(base)
    fig = go.Figure()
    for m in sorted(w["Machine_ID"].unique(), key=lambda m: m in TOP):
        s = w[w["Machine_ID"] == m]
        fig.add_scatter(x=s["Week"], y=s["Index"], name=m, mode="lines",
                        line=dict(color=machine_color(m), width=2.5 if m in TOP else 1),
                        opacity=1 if m in TOP else 0.6,
                        hovertemplate=f"{m}<br>%{{x|%d %b}}: %{{y:.1f}}<extra></extra>")
    fig.add_hline(y=100, line=dict(color="#52514e", width=1, dash="dot"))
    fig.add_vline(x=pd.Timestamp("2025-06-16"), line=dict(color=GRAY, width=1, dash="dash"))
    fig.update_yaxes(title="Index (baseline = 100)")
    show(fig, height=420)
    st.caption("The grey dashed line marks 16 Jun, when the overdue-maintenance share starts rising (see section 7). "
               "Output responds with a 4–6 week lag, as wear builds once machines pass the 30-day interval.")

    st.subheader("Monthly plant output")
    mo = daily.groupby(daily["Date"].dt.to_period("M").dt.to_timestamp())["Output"].mean().reset_index()
    fig = px.bar(mo, x="Date", y="Output", text="Output")
    fig.update_traces(marker_color=[BLUE if d >= pd.Timestamp(D0.year, D0.month, 1) else GRAY for d in mo["Date"]],
                      texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_yaxes(title="Mean good units / day", range=[mo["Output"].min() * 0.9, mo["Output"].max() * 1.03])
    fig.update_xaxes(title=None, dtick="M1", tickformat="%b")
    show(fig, legend=False)


# ============================================================================= 4. MACHINE LOSSES
def page_losses():
    st.title("4 · Machines responsible for the largest losses")
    st.markdown(f"**Loss** is each machine's expected good output after the decline began, based on its "
                f"pre-{D0:%d %b} average for the same shift, minus what it actually produced.")
    l, r = st.columns([3, 2])
    with l:
        fig_pareto()
    with r:
        st.dataframe(losses.round(1), hide_index=True, width="stretch")

    st.subheader("Why each machine lost output: split into OEE components")
    st.markdown("Good output = ideal rate × 480 min × **Availability × Performance × Quality**. The drop in each "
                "machine's log-output is split across the three factors.")
    c = comp[comp["Machine_ID"].isin(losses.loc[losses["Units_Lost"] > 0, "Machine_ID"])]
    order = losses["Machine_ID"].tolist()
    fig = px.bar(c, y="Machine_ID", x="Units_Lost", color="Component", orientation="h",
                 color_discrete_map=OEE_C, category_orders={"Machine_ID": order,
                                                            "Component": list(OEE_C)})
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)",
                      hovertemplate="%{y} · %{fullData.name}<br>%{x:,.0f} units<extra></extra>")
    fig.update_xaxes(title="Good units lost")
    fig.update_yaxes(title=None)
    show(fig, height=360)
    tot = comp.groupby("Component")["Units_Lost"].sum()
    st.caption(" · ".join(f"{k}: {100 * v / tot.sum():.0f}% of total loss" for k, v in tot.items()))

    st.subheader("Machine × month good output")
    mm = df.pivot_table(index="Machine_ID", columns="Month", values="Machine_Output", aggfunc="mean")
    mm = mm.div(mm.iloc[:, :5].mean(axis=1), axis=0) * 100
    mm.columns = [c.strftime("%b") for c in mm.columns]
    fig = px.imshow(mm.round(1), text_auto=True, aspect="auto", zmin=70, zmax=110,
                    color_continuous_scale=[[0, "#a8281f"], [0.75, "#f0efec"], [1, "#256abf"]])
    fig.update_layout(coloraxis_colorbar=dict(title="Index"))
    show(fig, height=380)
    st.caption("Index: 100 = the machine's own Jan–May average. Red cells = below baseline.")


# ============================================================================= 5. DOWNTIME
def page_downtime():
    st.title("5 · Downtime patterns")
    tot = df.groupby("Period")[["Planned_Downtime", "Unplanned_Downtime"]].mean()
    c = st.columns(4)
    c[0].metric("Unplanned downtime / shift, before", f"{tot.loc['Before decline', 'Unplanned_Downtime']:.1f} min")
    c[1].metric("Unplanned downtime / shift, after", f"{tot.loc['After decline', 'Unplanned_Downtime']:.1f} min",
                f"{tot.loc['After decline', 'Unplanned_Downtime'] - tot.loc['Before decline', 'Unplanned_Downtime']:+.1f} min",
                delta_color="inverse")
    c[2].metric("Planned downtime / shift, before", f"{tot.loc['Before decline', 'Planned_Downtime']:.1f} min")
    c[3].metric("Planned downtime / shift, after", f"{tot.loc['After decline', 'Planned_Downtime']:.1f} min",
                f"{tot.loc['After decline', 'Planned_Downtime'] - tot.loc['Before decline', 'Planned_Downtime']:+.1f} min",
                delta_color="off")
    st.info("Planned maintenance time **fell** while unplanned downtime **rose**. Skipping services did not free up "
            "capacity. The saved maintenance time came back as more, longer breakdowns.")

    st.subheader("Unplanned downtime heat-map (mean minutes per shift)")
    h = A.downtime_heatmap(df)
    h.columns = [c.strftime("%b") for c in h.columns]
    fig = px.imshow(h.round(0), text_auto=True, aspect="auto", color_continuous_scale=SEQ_BLUE)
    fig.update_layout(coloraxis_colorbar=dict(title="min"))
    show(fig, height=380)

    l, r = st.columns(2)
    with l:
        st.subheader("Weekly downtime split, plant total")
        wk = df.groupby("Week")[["Planned_Downtime", "Unplanned_Downtime"]].sum().div(60).reset_index()
        wk = wk.melt(id_vars="Week", var_name="Type", value_name="Hours")
        wk["Type"] = wk["Type"].str.replace("_Downtime", "")
        fig = px.bar(wk, x="Week", y="Hours", color="Type",
                     color_discrete_map={"Planned": GRAY, "Unplanned": RED})
        fig.update_layout(bargap=0.1)
        fig.update_yaxes(title="Hours / week")
        fig.update_xaxes(title=None)
        show(fig)
    with r:
        st.subheader("Breakdown rate by machine")
        br = df.groupby(["Machine_ID", "Period"])["Breakdown"].mean().mul(100).reset_index()
        fig = px.bar(br, x="Machine_ID", y="Breakdown", color="Period", barmode="group",
                     color_discrete_map=PERIOD, category_orders={"Period": ["Before decline", "After decline"]})
        fig.update_yaxes(title=f"% of shifts with > {A.UNPLANNED_BREAKDOWN_MIN} min unplanned stop")
        fig.update_xaxes(title=None)
        show(fig)

    l, r = st.columns(2)
    with l:
        st.subheader("By shift")
        s = A.downtime_summary(df, ["Shift", "Period"])
        fig = px.bar(s, x="Shift", y="Unplanned", color="Period", barmode="group", color_discrete_map=PERIOD,
                     category_orders={"Period": ["Before decline", "After decline"]}, text="Unplanned")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_yaxes(title="Unplanned min / shift")
        show(fig)
    with r:
        st.subheader("By day of week")
        s = A.downtime_summary(df, "DayOfWeek")
        fig = px.bar(s, x="DayOfWeek", y="Unplanned", text="Unplanned")
        fig.update_traces(marker_color=BLUE, texttemplate="%{text:.1f}", textposition="outside")
        fig.update_yaxes(title="Unplanned min / shift")
        fig.update_xaxes(title=None)
        show(fig, legend=False)
    st.caption("Downtime is similar across shifts and days of the week. It is concentrated by machine and by "
               "time since the last service, not by when the shift runs or who runs it.")

    st.subheader("Distribution of unplanned downtime")
    fig = px.histogram(df, x="Unplanned_Downtime", color="Period", nbins=80, barmode="overlay", opacity=0.65,
                       color_discrete_map=PERIOD, log_y=True)
    fig.update_xaxes(title="Unplanned downtime (min / shift)")
    fig.update_yaxes(title="Shifts (log scale)")
    show(fig)


# ============================================================================= 6. SHIFTS & OPERATORS
def page_people():
    st.title("6 · Shifts & operators: a fair comparison")
    st.warning("**How to read this section.** Operators are rostered to fixed shift and machine groups, so a raw "
               "average mostly reflects **which machine** someone runs and **when**, not how well they work. Every "
               "comparison below is therefore **adjusted**: each shift's output is compared with the mean for the "
               "*same machine* in the *same week* (and, for operators, the *same shift*). Operators are shown "
               "only by anonymised code. These results are about the process and must not be used to evaluate "
               "individual performance.")

    stbl, stest = shift_stats
    l, r = st.columns(2)
    with l:
        st.subheader("Shifts: raw vs adjusted")
        t = stbl.melt(id_vars="Shift", value_vars=["Raw_vs_Avg_%", "Adj_Mean"], var_name="Measure", value_name="%")
        t["Measure"] = t["Measure"].map({"Raw_vs_Avg_%": "Raw vs plant average",
                                         "Adj_Mean": "Adjusted (same machine & week)"})
        fig = px.bar(t, x="Shift", y="%", color="Measure", barmode="group", text="%",
                     color_discrete_sequence=[GRAY, BLUE])
        fig.update_traces(texttemplate="%{text:+.1f}%", textposition="outside")
        fig.update_yaxes(title="% difference in good output")
        show(fig)
    with r:
        st.subheader("Night-shift gap before vs after decline")
        adj = df.assign(Adj=A.adjusted_residuals(df))
        g = adj.groupby(["Period", "Shift"], observed=True)["Adj"].agg(["mean", "std", "size"]).reset_index()
        g["ci"] = 1.96 * g["std"] / np.sqrt(g["size"])
        fig = px.bar(g, x="Shift", y="mean", color="Period", barmode="group", error_y="ci",
                     color_discrete_map=PERIOD, category_orders={"Period": ["Before decline", "After decline"]})
        fig.update_yaxes(title="Adjusted % difference (±95% CI)")
        show(fig)
    st.markdown(f"Kruskal-Wallis across shifts: H = {stest['H']:.0f}, p = {stest['p']:.1e}, "
                f"**effect size ε² = {stest['eps2']:.3f}** (small). The night-shift gap is real but small, and it "
                "is **the same before and after the decline**, so it does not explain the decline. It may reflect "
                "fatigue or support levels at night, which is a process issue, not a people issue.")

    st.subheader("Operators: raw vs adjusted")
    otbl, otest = op_stats
    mix = A.operator_mix(df, TOP)
    o = otbl[otbl["Operator"] != "Unknown"].merge(mix, on="Operator")
    fig = go.Figure()
    fig.add_bar(x=o["Operator"], y=o["Raw_vs_Avg_%"], name="Raw vs plant average", marker_color=GRAY)
    fig.add_bar(x=o["Operator"], y=o["Adj_Mean"], name="Adjusted (same machine, week & shift)",
                marker_color=BLUE, error_y=dict(type="data", array=o["CI95"], thickness=1, width=3))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="% difference in good output")
    show(fig, height=400)
    st.markdown(f"Raw averages range from **{o['Raw_vs_Avg_%'].min():+.0f}%** to **{o['Raw_vs_Avg_%'].max():+.0f}%**. "
                f"After adjustment every operator is within **±{o['Adj_Mean'].abs().max():.1f}%** "
                f"(ε² = {otest['eps2']:.3f}). The raw gap comes from **machine assignment**: operators rostered to "
                f"the high-capacity presses ({', '.join(sorted(TOP))}) appear 'better' simply because those machines "
                "are faster.")
    st.dataframe(o[["Operator", "Main_Shift", "Shifts_Worked", "Share_on_Neglected_Machines",
                    "Raw_vs_Avg_%", "Adj_Mean", "CI95"]]
                 .rename(columns={"Share_on_Neglected_Machines": f"% shifts on {'/'.join(sorted(TOP))}",
                                  "Adj_Mean": "Adjusted_%", "CI95": "±95% CI"}).round(2),
                 hide_index=True, width="stretch")
    st.caption("Remaining small differences are within the range expected from random variation and unrecorded "
               "factors such as job mix or material batches. There is no evidence here to support conclusions "
               "about any individual.")


# ============================================================================= 7. MAINTENANCE
def page_maintenance():
    st.title("7 · Maintenance vs downtime")
    l, r = st.columns(2)
    with l:
        st.subheader("Downtime by days since maintenance")
        fig_maint_age()
    with r:
        st.subheader("Breakdown probability & temperature by maintenance age")
        b = maint["by_age"]
        fig = go.Figure()
        fig.add_scatter(x=b["Age_Bin"].astype(str), y=100 * b["Breakdown_Rate"], mode="lines+markers",
                        name="Breakdown rate (%)", line=dict(color=RED, width=2), marker=dict(size=9))
        fig.update_yaxes(title="% of shifts with a breakdown")
        fig.update_xaxes(title="Days since last planned maintenance", type="category")
        show(fig, legend=False)

    st.subheader("Downtime by recorded maintenance status")
    s = maint["by_status"].copy()
    s = s.set_index("Maintenance_Status").loc[["Up to Date", "Due Soon", "Overdue", "In Progress"]].reset_index()
    t = s.melt(id_vars="Maintenance_Status", value_vars=["Planned", "Unplanned"], var_name="Type", value_name="min")
    fig = px.bar(t, x="Maintenance_Status", y="min", color="Type",
                 color_discrete_map={"Planned": GRAY, "Unplanned": RED})
    fig.update_yaxes(title="Mean downtime (min / shift)")
    fig.update_xaxes(title=None)
    show(fig)
    st.dataframe(s.round(2), hide_index=True, width="stretch")
    st.markdown(f"Overdue vs up-to-date unplanned downtime: one-sided Mann-Whitney **p = {maint['mw_p']:.1e}**. "
                f"The pooled Spearman ρ between maintenance age and downtime is only {maint['rho']:.2f}, because "
                "most shifts have near-zero downtime whatever the maintenance age. The effect appears once a machine "
                "is well past its interval, as the binned chart above shows.")

    l, r = st.columns(2)
    with l:
        st.subheader("Share of shifts running overdue, by machine")
        od = maint["overdue_share"]
        fig = go.Figure()
        for m in sorted(od["Machine_ID"].unique(), key=lambda m: m in TOP):
            x = od[od["Machine_ID"] == m]
            fig.add_scatter(x=x["Month"], y=x["Overdue"], name=m, mode="lines+markers",
                            line=dict(color=machine_color(m), width=2.5 if m in TOP else 1),
                            marker=dict(size=8 if m in TOP else 5))
        fig.update_yaxes(title="% of shifts overdue")
        fig.update_xaxes(dtick="M1", tickformat="%b")
        show(fig)
    with r:
        st.subheader("Is it just summer heat? Seasonal control")
        sc = maint["season_check"].reset_index().melt(id_vars="Season", var_name="Status", value_name="min")
        fig = px.bar(sc, x="Season", y="min", color="Status", barmode="group", text="min",
                     color_discrete_map=STATUS)
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_yaxes(title="Unplanned min / shift")
        fig.update_xaxes(title=None)
        show(fig)
    st.caption("Within the same season, overdue machines always have far more unplanned downtime. The effect is "
               "larger in Oct–Dec (cooler months) because wear keeps building while maintenance stays frozen, so "
               "temperature is a symptom of wear, not the root cause.")

    st.subheader("Temperature vs maintenance age (M03, M05, M06)")
    d = maint["data"]
    d = d[d["Machine_ID"].isin(TOP)].sample(min(3000, len(d)), random_state=1)
    fig = px.scatter(d, x="Days_Since_Maint", y="Temperature", color="Machine_ID", opacity=0.5,
                     color_discrete_map={m: machine_color(m) for m in TOP})
    fig.update_traces(marker=dict(size=6))
    fig.update_xaxes(title="Days since last planned maintenance")
    fig.update_yaxes(title="Temperature (°C)")
    show(fig)


# ============================================================================= 8. OEE
def page_oee():
    st.title("8 · Overall Equipment Effectiveness (OEE)")
    st.markdown("""
| Component | Definition used |
|---|---|
| **Availability** | (480 − Downtime) / 480 |
| **Performance** | Production_Volume / (Ideal rate × Run time), where ideal rate = the machine's 99th-percentile demonstrated units per running minute |
| **Quality** | Machine_Output (good) / Production_Volume (total) |
| **OEE** | Availability × Performance × Quality |
""")
    fig_oee_pp()

    st.subheader("Monthly OEE components, plant average")
    mo = A.oee_table(df, "Month").melt(id_vars="Month", var_name="Metric", value_name="Value")
    l, r = st.columns([2, 1])
    with l:
        fig = px.line(mo[mo["Metric"] != "OEE"], x="Month", y="Value", color="Metric", markers=True,
                      color_discrete_map=OEE_C)
        fig.update_traces(line=dict(width=2), marker=dict(size=8))
        fig.update_yaxes(title="%")
        fig.update_xaxes(dtick="M1", tickformat="%b", title=None)
        fig.update_layout(hovermode="x unified")
        show(fig)
    with r:
        o = mo[mo["Metric"] == "OEE"]
        fig = px.bar(o, x="Month", y="Value", text="Value")
        fig.update_traces(marker_color=[BLUE if m >= pd.Timestamp(D0.year, D0.month, 1) else GRAY
                                        for m in o["Month"]],
                          texttemplate="%{text:.0f}", textposition="outside")
        fig.update_yaxes(title="OEE %", range=[70, 95])
        fig.update_xaxes(dtick="M1", tickformat="%b", title=None)
        show(fig, legend=False)

    st.subheader("OEE by machine, before vs after decline")
    t = A.oee_table(df, ["Machine_ID", "Period"])
    wide = t.pivot(index="Machine_ID", columns="Period")
    wide.columns = [f"{a} ({'pre' if b == 'Before decline' else 'post'})" for a, b in wide.columns]
    for k in ["Availability", "Performance", "Quality", "OEE"]:
        wide[f"Δ {k}"] = wide[f"{k} (post)"] - wide[f"{k} (pre)"]
    wide = wide.sort_values("Δ OEE")
    fig = go.Figure()
    fig.add_bar(y=wide.index, x=wide["Δ OEE"], orientation="h",
                marker_color=[RED if v < -2 else GRAY for v in wide["Δ OEE"]],
                text=[f"{v:+.1f}" for v in wide["Δ OEE"]], textposition="outside")
    fig.update_xaxes(title="Change in OEE (percentage points)")
    show(fig, height=340, legend=False)
    st.dataframe(wide.round(1), width="stretch")


# ============================================================================= 9. MODEL
def page_model():
    st.title("9 · Random Forest model")
    st.markdown("""
**Goal:** predict a shift's **good output** (`Machine_Output`) from conditions known *before* the shift runs,
then use the model to measure **which factors drive output** and **what maintenance delays cost**.

* **Features:** Machine_ID, Shift, Operator, Maintenance_Status, Day of week, Temperature, Days since maintenance
* **Excluded (leakage):** Downtime, Production_Volume and OEE components. These are measured during the shift and
  would give the answer away.
* **Model:** `RandomForestRegressor` (300 trees, max_depth 14, min_samples_leaf 5), with one-hot encoding in a pipeline.
* **Evaluation:** machine-stratified 80/20 hold-out + 5-fold cross-validation, compared with a naive machine × shift average.
""")
    with st.spinner("Training…"):
        res = train(df)
    rf, bl, trm = res["rf_metrics"], res["baseline_metrics"], res["train_metrics"]
    c = st.columns(4)
    c[0].metric("Hold-out R²", f"{rf['R²']:.3f}", f"{rf['R²'] - bl['R²']:+.3f} vs baseline")
    c[1].metric("Hold-out MAE", f"{rf['MAE']:.1f} units", f"{rf['MAE'] - bl['MAE']:+.1f} vs baseline",
                delta_color="inverse")
    c[2].metric("5-fold CV R²", f"{res['cv_r2'].mean():.3f}", f"± {res['cv_r2'].std():.3f}", delta_color="off")
    c[3].metric("Train / test rows", f"{res['n_train']:,} / {res['n_test']:,}")

    mt = pd.DataFrame({"Random Forest (test)": rf, "Naive baseline (test)": bl, "Random Forest (train)": trm}).T
    st.dataframe(mt.round(3), width="stretch")
    st.caption("The small train/test gap (R² {:.2f} vs {:.2f}) means the model is not badly overfitting. "
               "The remaining error mostly comes from random breakdowns that no pre-shift feature can "
               "predict exactly.".format(trm["R²"], rf["R²"]))

    l, r = st.columns(2)
    with l:
        st.subheader("Permutation importance (test set)")
        imp = res["importance"].sort_values("Importance")
        fig = go.Figure(go.Bar(y=imp["Feature"], x=imp["Importance"], orientation="h",
                               error_x=dict(type="data", array=imp["Std"], thickness=1),
                               marker_color=[BLUE if f in ("Days_Since_Maint", "Maintenance_Status") else GRAY
                                             for f in imp["Feature"]]))
        fig.update_xaxes(title="Increase in MAE when shuffled (units)")
        show(fig, legend=False)
        st.caption("Machine_ID ranks first because the machines have different rated speeds. Among the factors "
                   "the plant can change, **maintenance age matters most**. Operator, shift and day of week add "
                   "almost nothing, which agrees with section 6. Temperature adds little once maintenance age is "
                   "known, because it is a symptom of wear.")
    with r:
        st.subheader("Actual vs predicted (test set)")
        t = res["test"]
        fig = px.scatter(t, x="Predicted", y="Machine_Output", color="Maintenance_Status", opacity=0.55,
                         color_discrete_map=STATUS)
        lim = [t[["Predicted", "Machine_Output"]].min().min(), t[["Predicted", "Machine_Output"]].max().max()]
        fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color="#52514e", dash="dot", width=1),
                        name="Perfect prediction")
        fig.update_traces(selector=dict(mode="markers"), marker=dict(size=6))
        fig.update_yaxes(title="Actual good output")
        show(fig)

    st.subheader("What-if: cost of delaying maintenance")
    m = st.selectbox("Machine", sorted(df["Machine_ID"].unique()), index=sorted(df["Machine_ID"].unique()).index(TOP[0]))
    sh = st.radio("Shift", ["Morning", "Afternoon", "Night"], horizontal=True)
    row = df[(df["Machine_ID"] == m) & (df["Shift"] == sh)].iloc[[0]].copy()
    row["Temperature"] = df.loc[df["Machine_ID"] == m, "Temperature"].median()
    ages = list(range(0, 181, 10))
    wi = M.what_if(res["model"], row, ages)
    fig = px.line(wi, x="Days_Since_Maint", y="Predicted_Output", markers=True)
    fig.update_traces(line=dict(color=BLUE, width=2), marker=dict(size=8))
    fig.add_vline(x=30, line=dict(color="#52514e", dash="dot", width=1), annotation_text="30-day interval")
    fig.update_xaxes(title="Days since last maintenance")
    fig.update_yaxes(title="Predicted good units / shift")
    show(fig, legend=False)
    drop = 100 * (wi["Predicted_Output"].iloc[-1] / wi["Predicted_Output"].iloc[0] - 1)
    st.markdown(f"For **{m}** on the {sh} shift, the model predicts **{drop:+.0f}%** output at 180 days without service "
                f"compared with a freshly serviced machine. Temperature is held at the machine median.")

    with st.expander("Limitation: forward-in-time test (train Jan–Sep, test Oct–Dec)"):
        f = res["forward"]
        st.markdown(f"""
Trained on Jan–Sep and tested on Oct–Dec, the model scores R² = **{f['rf']['R²']:.2f}** (baseline {f['baseline']['R²']:.2f}),
MAE {f['rf']['MAE']:.1f} vs {f['baseline']['MAE']:.1f}. It still beats the baseline, but R² is poor because in
Oct–Dec the frozen machines reached **{f['max_age_test']:.0f} days** without service, while training only saw up to
**{f['max_age_train']:.0f} days**. **Random Forests cannot extrapolate beyond the range they were trained on**, so the
model under-predicts how bad things get. To use it operationally, retrain regularly, and above all keep machines inside
the service window the model has learned from.
""")


# ============================================================================= ACTION PLAN
def page_action():
    st.title("Practical action plan")
    L = losses.set_index("Machine_ID")
    st.markdown(f"""
### Root cause
From mid-June, planned maintenance on **{TOP[0]} and {TOP[1]}** stopped, and service intervals on **{TOP[2]}**
were stretched. Once past the 30-day interval these machines ran hotter, broke down more often, ran slower and
scrapped more parts. Plant output started falling around **{D0:%d %b}**. These three machines account for
**{losses.head(3)['Share_of_Loss_%'].sum():.0f}%** of the lost output. Shifts, operators and summer heat are **not**
the cause.

### Actions
| # | Action | Owner | When | Target KPI |
|---|---|---|---|---|
| 1 | **Service {TOP[0]} and {TOP[1]} immediately** (overhaul: bearings, lubrication, cooling), then {TOP[2]} | Maintenance lead | Next 2 weeks | Unplanned downtime on these machines < 20 min/shift |
| 2 | **Restore the 30-day preventive maintenance interval on all machines.** Make overdue maintenance a stop-the-line escalation, not a budget line to cut | Plant manager | This month | 0 machines > 35 days since service |
| 3 | **Business case:** ≈{(L.loc[TOP[0], 'Units_Lost'] + L.loc[TOP[1], 'Units_Lost']) / max((df['Date'].max() - D0).days, 1):,.0f} good units/day lost on {TOP[0]} and {TOP[1]} alone, far more than the cost of the maintenance hours saved | Finance + Ops | This month | Maintenance budget restored |
| 4 | **Condition-based alerts:** flag any machine whose temperature is more than 5 °C above its own baseline, or that passes 30 days since service | Controls / IT | 1–2 months | Alerts reviewed at the daily stand-up |
| 5 | **Use the Random Forest model weekly** to rank machines by predicted output loss and schedule maintenance before breakdowns | Data / Maintenance | 1–2 months | Hold-out R² ≥ 0.7, retrained monthly |
| 6 | **Track OEE daily by machine** (A × P × Q) on the shop-floor dashboard, and treat availability drops as the leading signal | Production supervisors | Ongoing | Plant OEE back to ≥ {oee_pp.set_index('Period').loc['Before decline', 'OEE']:.0f}% |
| 7 | **Night shift (small, constant −1% gap):** review support at night (maintenance cover, lighting, breaks) as a process question. Do **not** make individual performance judgements from this data | HR + Production | Quarterly | Gap closed without blame |
| 8 | **Improve data quality at source:** a single date format, validated drop-downs for shift and machine, sensor range checks, and a de-duplicated upload | IT | 1 month | < 0.5% records need cleaning |

### Expected impact
Bringing the three affected machines back to their baselines recovers about **{losses.head(3)['Units_Lost'].sum() / max((df['Date'].max() - D0).days, 1):,.0f}
good units per day**, which restores plant output to the pre-decline level of about
**{df[df['Period'] == 'Before decline'].groupby('Date')['Machine_Output'].sum().mean():,.0f} units/day**.
""")
    st.caption("Note: the dataset is synthetic and was generated for this exercise (see generate_data.py). "
               "The methods carry over directly to real plant data.")


# ----------------------------------------------------------------------------- router
{
    PAGES[0]: page_overview, PAGES[1]: page_cleaning, PAGES[2]: page_eda, PAGES[3]: page_decline,
    PAGES[4]: page_losses, PAGES[5]: page_downtime, PAGES[6]: page_people, PAGES[7]: page_maintenance,
    PAGES[8]: page_oee, PAGES[9]: page_model, PAGES[10]: page_action,
}[page]()
