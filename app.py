"""PETE 662 Project 2 - Interactive Nodal Analysis and Sensitivity Tool.

Streamlit front end for the ``pete662_nodal`` package.

    Run from this folder:   streamlit run app.py

Carlos A. Buenrostro, Texas A&M University, Fall 2026.
"""
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from pete662_nodal import (WellboreCase, IPRInputs, ReservoirInputs, oil_rate, ipr_table,
                           pressure_traverse, operating_point, run_sensitivity, make_pvt)
from pete662_nodal.traverse import METHOD_NAMES

# --------------------------------------------------------------------------
# Page setup and styling
# --------------------------------------------------------------------------
st.set_page_config(page_title="PETE 662 Nodal Analysis", page_icon="🛢️", layout="wide")

MAROON = "#500000"
VLP_COLORS = {"HB": "#1f5fa8", "BB": "#d9822b"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

# --------------------------------------------------------------------------
# Example cases (loading one resets every input to its values)
# --------------------------------------------------------------------------
EXAMPLES = {
    "Exam 1 Problem 1 (PPS Appendix B oil, saturated)": dict(
        ipr_mode="Reservoir properties", pr=3000.0, pb=4336.0, J=1.0,
        k=13.0, kro=0.5, h=115.0, mu=0.5, Bo=1.46, re=1490.0, rw=0.25, skin=0.0, pss=True,
        depth=9000.0, ptf=100.0, Ts=150.0, Tb=180.0, D_in=2.44, rough=0.0006, theta=90.0, dz=200.0,
        GOR=750.0, WOR=0.0, API=32.0, gg=0.71, Rsb=800.0, sigma=30.0, mu_g=0.015, mu_w=0.7,
        pvt="appendix_b"),
    "Undersaturated oil well with water cut (correlations)": dict(
        ipr_mode="Productivity index J", pr=4500.0, pb=2450.0, J=1.0,
        k=20.0, kro=1.0, h=60.0, mu=0.8, Bo=1.3, re=1490.0, rw=0.328, skin=0.0, pss=True,
        depth=8000.0, ptf=150.0, Ts=100.0, Tb=200.0, D_in=2.441, rough=0.0006, theta=90.0, dz=200.0,
        GOR=500.0, WOR=0.25, API=35.0, gg=0.75, Rsb=500.0, sigma=30.0, mu_g=0.015, mu_w=0.7,
        pvt="correlations"),
}
TUBING = {  # label -> ID in inches (API tubing)
    "2-3/8 in, 4.7 lb/ft (ID 1.995 in)": 1.995,
    "2-7/8 in, 8.6 lb/ft (ID 2.259 in)": 2.259,
    "2-7/8 in, 6.5 lb/ft (ID 2.441 in)": 2.441,
    "3-1/2 in, 9.3 lb/ft (ID 2.992 in)": 2.992,
    "4-1/2 in, 12.75 lb/ft (ID 3.958 in)": 3.958,
    "Custom ID": None,
}
PVT_LABELS = {"correlations": "Textbook correlations (Ch. 3-4)",
              "appendix_b": "Appendix-B curves (HW5 / Exam 1 fluid)"}
METHOD_CHOICES = {"Modified Hagedorn-Brown": ["HB"], "Beggs-Brill (Payne)": ["BB"],
                  "Both (compare)": ["HB", "BB"]}


# --------------------------------------------------------------------------
# Cached calculations (inputs are frozen dataclasses, so they hash cleanly)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=200)
def cached_nodal(inflow, case, method):
    return operating_point(inflow, case, method)


@st.cache_data(show_spinner=False, max_entries=200)
def cached_profile(q, case, method):
    return pressure_traverse(q, case, method, return_profile=True)


@st.cache_data(show_spinner=False, max_entries=50)
def cached_sensitivity(inflow, case, name, values, method):
    return run_sensitivity(inflow, case, name, list(values), method)


# --------------------------------------------------------------------------
# Sidebar - inputs
# --------------------------------------------------------------------------
st.sidebar.title("Inputs")
example = st.sidebar.selectbox("Load example case", list(EXAMPLES), index=0,
                               help="Loading an example resets all inputs below.")
E = EXAMPLES[example]
K = lambda name: f"{example}|{name}"          # widget keys reset when the example changes


def num(label, key, minv=None, maxv=None, step=None, fmt=None, help=None):
    return st.number_input(label, value=float(E[key]), min_value=minv, max_value=maxv,
                           step=step, format=fmt, key=K(key), help=help)


with st.sidebar.expander("1 · Reservoir / inflow (IPR)", expanded=True):
    ipr_mode = st.radio("Specify inflow by", ["Reservoir properties", "Productivity index J"],
                        index=0 if E["ipr_mode"] == "Reservoir properties" else 1, key=K("ipr_mode"))
    pr = num("Average reservoir pressure, p̄ (psia)", "pr", 1.0, step=50.0)
    pb = num("Bubble-point pressure, pb (psia)", "pb", 1.0, step=50.0,
             help="Used by the IPR: PI above pb, (generalized) Vogel below.")
    if ipr_mode == "Productivity index J":
        J = num("Productivity index, J (STB/d/psi)", "J", 0.001, step=0.1, fmt="%.3f",
                help="If p̄ ≤ pb this is the tangent PI at p̄ (Vogel).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            k = num("k (md)", "k", 0.001, step=1.0)
            h = num("h (ft)", "h", 0.1, step=5.0)
            mu = num("μo (cp)", "mu", 0.01, step=0.05)
            re_ = num("re (ft)", "re", 1.0, step=50.0)
            skin = num("Skin, s", "skin", -6.0, step=1.0)
        with c2:
            kro = num("kro", "kro", 0.001, 1.0, step=0.05)
            Bo = num("Bo (bbl/STB)", "Bo", 1.0, step=0.01)
            rw = num("rw (ft)", "rw", 0.01, step=0.01)
            pss = st.checkbox("Pseudo-steady state (−0.75)", value=E["pss"], key=K("pss"))

with st.sidebar.expander("2 · Well and tubing", expanded=False):
    depth = num("Depth to node / mid-perfs (ft)", "depth", 100.0, step=100.0)
    ptf = num("Flowing tubing-head pressure (psia)", "ptf", 14.7, step=10.0)
    c1, c2 = st.columns(2)
    with c1:
        Ts = num("Surface T (°F)", "Ts", 32.0, step=5.0)
    with c2:
        Tb = num("Bottomhole T (°F)", "Tb", 32.0, step=5.0)
    tub_labels = list(TUBING)
    default_tub = next((lbl for lbl, d in TUBING.items() if d == E["D_in"]), "Custom ID")
    tub = st.selectbox("Tubing", tub_labels, index=tub_labels.index(default_tub), key=K("tub"))
    D_in = TUBING[tub] if TUBING[tub] else num("Tubing ID (in)", "D_in", 0.5, 8.0, step=0.01, fmt="%.3f")
    rough = num("Relative roughness, ε/D", "rough", 0.0, 0.05, step=0.0001, fmt="%.4f")
    theta = num("Inclination from horizontal (°)", "theta", 1.0, 90.0, step=5.0,
                help="90° = vertical. Used by Beggs-Brill; modified H-B is for near-vertical wells.")
    dz = num("Traverse step, Δz (ft)", "dz", 10.0, 1000.0, step=50.0)

with st.sidebar.expander("3 · Production and fluid", expanded=False):
    GOR = num("Producing GOR (scf/STB)", "GOR", 0.0, step=25.0)
    WOR = num("Water-oil ratio, WOR (STB/STB)", "WOR", 0.0, step=0.1)
    c1, c2 = st.columns(2)
    with c1:
        API = num("Oil gravity (°API)", "API", 5.0, 70.0, step=1.0)
        sigma = num("σL (dyn/cm)", "sigma", 1.0, step=1.0)
    with c2:
        gg = num("Gas gravity (air = 1)", "gg", 0.55, 1.5, step=0.01)
        mu_w = num("μw (cp)", "mu_w", 0.1, step=0.1)
    pvt_model = st.radio("PVT model", list(PVT_LABELS), format_func=PVT_LABELS.get,
                         index=list(PVT_LABELS).index(E["pvt"]), key=K("pvt"))
    if pvt_model == "appendix_b":
        Rsb = num("Rsb for Appendix-B curves (scf/STB)", "Rsb", 1.0, step=25.0)
        mu_g = num("Gas viscosity (cp, constant)", "mu_g", 0.001, step=0.001, fmt="%.4f")
    else:
        Rsb, mu_g = E["Rsb"], E["mu_g"]

with st.sidebar.expander("4 · Calculation options", expanded=True):
    method_label = st.radio("Vertical-lift (VLP) correlation", list(METHOD_CHOICES), index=2,
                            key=K("method"))
    methods = METHOD_CHOICES[method_label]
    friction = st.radio("Friction factor", ["haaland", "chen"], horizontal=True, key=K("fric"),
                        format_func={"haaland": "Haaland", "chen": "Chen (Eq. 7-35)"}.get)

# --------------------------------------------------------------------------
# Build input objects
# --------------------------------------------------------------------------
try:
    if ipr_mode == "Productivity index J":
        inflow = IPRInputs(pr, pb, J)
    else:
        inflow = ReservoirInputs(pr, pb, k_md=k, h_ft=h, mu_o_cp=mu, Bo=Bo, re_ft=re_, rw_ft=rw,
                                 skin=skin, kro=kro, pseudo_steady=pss)
    ipr = inflow.to_ipr()
    case = WellboreCase(depth=depth, ptf=ptf, T_surface=Ts, T_bottom=Tb, D_in=D_in,
                        rel_roughness=rough, theta_deg=theta, dz=dz, friction_method=friction,
                        pvt_model=pvt_model, GOR=GOR, WOR=WOR, API=API, gamma_g=gg, pb=pb, Rsb=Rsb,
                        sigma_L=sigma, mu_g=mu_g, mu_w=mu_w)
except ValueError as err:
    st.error(f"Input problem: {err}")
    st.stop()

qmax = oil_rate(0.0, ipr)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(f"<h2 style='color:{MAROON};margin-bottom:0'>Interactive Nodal Analysis — Oil Well</h2>",
            unsafe_allow_html=True)
st.caption("PETE 662 Production Engineering · Texas A&M University · Carlos A. Buenrostro  —  "
           "IPR (PI / Vogel) + VLP (modified Hagedorn-Brown, Beggs-Brill) per "
           "*Petroleum Production Systems*, Ch. 2-3, 7, 9.")

warnings = []
if "HB" in methods and theta < 80:
    warnings.append("Modified Hagedorn-Brown is recommended only for near-vertical wells "
                    f"(inclination = {theta:.0f}°). Prefer Beggs-Brill.")
if pvt_model == "correlations":
    pb_fluid = make_pvt(case).bubble_point(Tb)
    if abs(pb_fluid - pb) / pb > 0.15:
        warnings.append(f"PVT check: Vasquez-Beggs puts the produced-fluid bubble point at "
                        f"{pb_fluid:,.0f} psia ({Tb:.0f} °F), but the IPR uses pb = {pb:,.0f} psia.")
for w in warnings:
    st.warning(w)

with st.spinner("Solving IPR / VLP intersection…"):
    results = {m: cached_nodal(inflow, case, m) for m in methods}
primary = methods[0]
res = results[primary]

tab_nodal, tab_profile, tab_sens, tab_data, tab_about = st.tabs(
    ["📈 Nodal analysis", "🌡️ Pressure profile", "🔁 Sensitivity", "📋 Data & report", "ℹ️ Method"])


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def show(fig, frac=0.68):
    """Draw a figure in the left `frac` of the page so it keeps a readable size on
    wide screens (Streamlit otherwise stretches it to the full width).
    Returns the right-hand column for side content."""
    left, right = st.columns([frac, 1 - frac])
    with left:
        st.pyplot(fig)
    plt.close(fig)
    return right


def nodal_figure(ipr_obj, res_by_method, title, extra_ipr=None):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    t = ipr_table(ipr_obj, 200)
    ax.plot(t.qo_stb_day, t.pwf_psia, color=MAROON, lw=2.4, label="IPR (inflow)")
    for m, r in res_by_method.items():
        ax.plot(r.table["qo (STB/d)"], r.table["VLP pwf (psia)"], color=VLP_COLORS[m], lw=2,
                label=f"VLP — {METHOD_NAMES[m]}")
        if r.status == "flowing":
            ax.scatter([r.q], [r.pwf], s=90, color=VLP_COLORS[m], edgecolor="k", zorder=5)
            offset = {"HB": (14, -38), "BB": (-92, 14)}[m]
            ax.annotate(f"{r.q:,.0f} STB/d\n{r.pwf:,.0f} psia", (r.q, r.pwf),
                        textcoords="offset points", xytext=offset, fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=VLP_COLORS[m], alpha=0.9))
        for qu, pu in r.unstable_points:
            ax.scatter([qu], [pu], s=60, facecolor="none", edgecolor=VLP_COLORS[m], zorder=5)
    if ipr_obj.bubblepoint_pressure < ipr_obj.reservoir_pressure:
        ax.axhline(ipr_obj.bubblepoint_pressure, ls=":", color="gray", lw=1)
        ax.text(0.99 * oil_rate(0, ipr_obj), ipr_obj.bubblepoint_pressure, " pb", va="bottom",
                ha="right", color="gray", fontsize=9)
    ax.set(xlabel="Oil rate, qo (STB/d)", ylabel="Flowing bottomhole pressure, pwf (psia)",
           title=title, xlim=(0, None), ylim=(0, None))
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Tab 1 - nodal analysis
# --------------------------------------------------------------------------
with tab_nodal:
    c = st.columns(4)
    if res.status == "flowing":
        c[0].metric(f"Oil rate ({primary})", f"{res.q:,.0f} STB/d")
        c[1].metric("Flowing BHP, pwf", f"{res.pwf:,.0f} psia")
    else:
        c[0].metric(f"Oil rate ({primary})", "—")
        c[1].metric("Flowing BHP, pwf", "—")
    c[2].metric("Productivity index, J", f"{ipr.productivity_index:.3f} STB/d/psi")
    c[3].metric("IPR maximum rate (AOF)", f"{qmax:,.0f} STB/d")

    for m, r in results.items():
        if r.status == "flowing":
            st.success(f"**{METHOD_NAMES[m]}:** oil {r.q:,.1f} STB/d  ·  liquid {r.q * (1 + WOR):,.1f} STB/d  ·  "
                       f"pwf = {r.pwf:,.1f} psia  ·  gas {r.q * GOR / 1000:,.0f} Mscf/d  ·  "
                       f"water {r.q * WOR:,.0f} STB/d")
        else:
            st.error(f"**{METHOD_NAMES[m]}:** {r.message}")
        if r.unstable_points:
            st.caption(f"{METHOD_NAMES[m]}: unstable (loading-branch) crossing at "
                       + ", ".join(f"{q:,.0f} STB/d" for q, _ in r.unstable_points)
                       + " — shown as a hollow marker, not an operating point.")
    if len(results) == 2 and all(r.status == "flowing" for r in results.values()):
        a, b = results["HB"], results["BB"]
        st.caption(f"Correlation difference: Δq = {b.q - a.q:+,.0f} STB/d "
                   f"({100 * (b.q - a.q) / a.q:+.1f} %), Δpwf = {b.pwf - a.pwf:+,.0f} psi. "
                   "Differences are reported, not forced to agree.")

    side = show(nodal_figure(ipr, results, "IPR / VLP — operating point at the bottomhole node"), frac=0.55)
    with side:
        st.markdown("**Operating point summary**")
        rows = []
        for m, r in results.items():
            rows.append({"VLP correlation": METHOD_NAMES[m],
                         "qo (STB/d)": f"{r.q:,.0f}" if r.q else "no flow",
                         "qL = qo + qw (STB/d)": f"{r.q * (1 + WOR):,.0f}" if r.q else "—",
                         "pwf (psia)": f"{r.pwf:,.0f}" if r.pwf else "—",
                         "Drawdown (psi)": f"{pr - r.pwf:,.0f}" if r.pwf else "—",
                         "qg (Mscf/d)": f"{r.q * GOR / 1000:,.0f}" if r.q else "—"})
        st.dataframe(pd.DataFrame(rows).set_index("VLP correlation").T)
        st.markdown("**Well**")
        st.markdown(f"Tubing ID {D_in:.3f} in · depth {depth:,.0f} ft  \n"
                    f"Wellhead {ptf:,.0f} psia · GOR {GOR:,.0f} scf/STB · WOR {WOR:g}  \n"
                    f"PVT: {PVT_LABELS[pvt_model]}")

# --------------------------------------------------------------------------
# Tab 2 - pressure profile
# --------------------------------------------------------------------------
with tab_profile:
    c1, c2 = st.columns([1, 1])
    with c1:
        prof_method = st.selectbox("Correlation", methods, format_func=METHOD_NAMES.get, key="prof_m")
    r0 = results[prof_method]
    default_q = float(round(r0.q)) if r0.status == "flowing" else float(round(0.5 * qmax))
    with c2:
        q_prof = st.number_input("Oil rate for the traverse (STB/d)", min_value=1.0,
                                 value=default_q, step=50.0, key=f"prof_q|{default_q}",
                                 help="Defaults to the operating rate.")
    pwf_p, prof = cached_profile(q_prof, case, prof_method)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True,
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.plot([case.ptf] + list(prof["P_out psia"]), [0.0] + list(prof["Depth ft"]),
             color=VLP_COLORS[prof_method], lw=2)
    ax1.plot([case.ptf, pwf_p], [0, case.depth], "o", color=VLP_COLORS[prof_method])
    pvt = make_pvt(case)
    if pvt_model == "correlations":
        pb_T = [pvt.bubble_point(case.T_at_depth(z)) for z in prof["Depth ft"]]
        ax1.plot(pb_T, prof["Depth ft"], ls=":", color="gray", label="Produced-fluid bubble point")
        ax1.legend(loc="lower left")
    ax1.set(xlabel="Pressure (psia)", ylabel="Depth (ft)", title=f"Pressure traverse at {q_prof:,.0f} STB/d")
    ax1.invert_yaxis()
    for regime, grp in prof.groupby("Regime"):
        ax2.scatter(grp["HL"], grp["Depth ft"], s=14, label=regime)
    ax2.set(xlabel="Liquid holdup, HL", title="Holdup and flow regime", xlim=(0, 1.02))
    ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    show(fig, frac=0.62)

    c = st.columns(3)
    c[0].metric("Bottomhole pressure", f"{pwf_p:,.1f} psia")
    c[1].metric("Average gradient", f"{(pwf_p - case.ptf) / case.depth:.4f} psi/ft")
    c[2].metric("Increments", f"{len(prof)} × {case.dz:.0f} ft")
    st.caption("Per-step values: pressures, gradient, holdup and regime, superficial and mixture velocities, "
               "no-slip liquid fraction, phase and mixture densities, and the Reynolds number used for friction.")
    st.dataframe(prof.round(4), height=300)
    st.download_button("Download traverse (CSV)", prof.to_csv(index=False),
                       f"traverse_{prof_method}_{q_prof:.0f}.csv", "text/csv")

# --------------------------------------------------------------------------
# Tab 3 - sensitivity
# --------------------------------------------------------------------------
SENS_VARS = {  # label -> (field, default values, applies-to)
    "Tubing ID (in)": ("D_in", "1.995, 2.441, 2.992, 3.958", "case"),
    "Wellhead pressure (psia)": ("ptf", f"{max(ptf * 0.5, 15):.0f}, {ptf:.0f}, {ptf * 2:.0f}, {ptf * 4:.0f}", "case"),
    "Producing GOR (scf/STB)": ("GOR", f"{GOR * 0.5:.0f}, {GOR:.0f}, {GOR * 1.5:.0f}, {GOR * 2:.0f}", "case"),
    "Water-oil ratio (STB/STB)": ("WOR", "0, 0.25, 0.5, 1.0", "case"),
    "Reservoir pressure (psia)": ("reservoir_pressure",
                                  f"{pr * 0.8:.0f}, {pr * 0.9:.0f}, {pr:.0f}, {pr * 1.1:.0f}", "both"),
}
if ipr_mode == "Productivity index J":
    SENS_VARS["Productivity index J (STB/d/psi)"] = ("productivity_index",
                                                     f"{J * 0.5:.2f}, {J:.2f}, {J * 1.5:.2f}, {J * 2:.2f}", "ipr")
else:
    SENS_VARS["Skin, s"] = ("skin", "-2, 0, 5, 10, 20", "res")
    SENS_VARS["Permeability, k (md)"] = ("k_md", f"{k * 0.5:g}, {k:g}, {k * 2:g}, {k * 4:g}", "res")

with tab_sens:
    st.markdown("Vary **one** input over a range while all other inputs stay at their sidebar values.")
    c1, c2, c3 = st.columns([1.2, 2, 1])
    with c1:
        sens_label = st.selectbox("Variable", list(SENS_VARS), key="sens_var")
    field, default_vals, _ = SENS_VARS[sens_label]
    with c2:
        vals_txt = st.text_input("Values (comma-separated)", value=default_vals,
                                 key=f"sens_vals|{sens_label}|{default_vals}")
    with c3:
        sens_method = st.selectbox("Correlation", methods, format_func=METHOD_NAMES.get, key="sens_m")
    go = st.button("Run sensitivity", type="primary")

    if go:
        try:
            values = tuple(float(v) for v in vals_txt.replace(";", ",").split(",") if v.strip())
            if not values:
                raise ValueError("enter at least one value")
            with st.spinner(f"Running {len(values)} nodal solutions…"):
                summary, runs = cached_sensitivity(inflow, case, field, values, sens_method)
            st.session_state["sens"] = dict(label=sens_label, field=field, method=sens_method,
                                            summary=summary, runs=runs, inflow=inflow, case=case)
        except ValueError as err:
            st.error(f"Could not run sensitivity: {err}")

    S = st.session_state.get("sens")
    if S:
        if S["inflow"] != inflow or S["case"] != case:
            st.info("Sidebar inputs changed since this study was run — press **Run sensitivity** to update.")
        summary, runs, field = S["summary"], S["runs"], S["field"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
        cmap = plt.get_cmap("viridis")
        n = len(runs)
        ipr_varies = field in ("skin", "k_md", "reservoir_pressure", "productivity_index")
        base_inflow = S["inflow"]
        for i, (v, r) in enumerate(zip(summary[field], runs)):
            col = cmap(i / max(n - 1, 1) * 0.9)
            ax1.plot(r.table["qo (STB/d)"], r.table["VLP pwf (psia)"], color=col, lw=1.8,
                     ls="-" if not ipr_varies else "--", label=f"{field} = {v:g}" if not ipr_varies else None)
            if ipr_varies:
                ipr_i = replace(base_inflow, **{field: v}).to_ipr()
                t = ipr_table(ipr_i, 150)
                ax1.plot(t.qo_stb_day, t.pwf_psia, color=col, lw=2, label=f"{field} = {v:g}")
            if r.status == "flowing":
                ax1.scatter([r.q], [r.pwf], color=col, edgecolor="k", s=60, zorder=5)
        if not ipr_varies:
            t = ipr_table(base_inflow.to_ipr(), 200)
            ax1.plot(t.qo_stb_day, t.pwf_psia, color=MAROON, lw=2.4, label="IPR")
        ax1.set(xlabel="Oil rate (STB/d)", ylabel="pwf (psia)", xlim=(0, None), ylim=(0, None),
                title=f"{S['label']} — {METHOD_NAMES[S['method']]}")
        ax1.legend(fontsize=8)
        ok = summary[summary["status"] == "flowing"]
        ax2.plot(ok[field], ok["q (STB/d)"], "o-", color=MAROON)
        ax2.set(xlabel=S["label"], ylabel="Operating oil rate (STB/d)", title="Operating rate")
        ax2b = ax2.twinx()
        ax2b.plot(ok[field], ok["pwf (psia)"], "s--", color="gray")
        ax2b.set_ylabel("pwf (psia)", color="gray")
        ax2b.spines["right"].set_visible(True)
        fig.tight_layout()
        show(fig, frac=0.75)
        st.dataframe(summary.round(3))
        st.download_button("Download sensitivity table (CSV)", summary.to_csv(index=False),
                           file_name=f"sensitivity_{field}.csv", mime="text/csv")

# --------------------------------------------------------------------------
# Tab 4 - data tables and example-case report
# --------------------------------------------------------------------------
def build_report():
    lines = [f"PETE 662 Nodal Analysis - example case report ({datetime.now():%Y-%m-%d %H:%M})",
             f"Example loaded: {example}", "", "INPUTS"]
    if isinstance(inflow, ReservoirInputs):
        lines += [f"  Reservoir pressure {pr:g} psia, bubble point {pb:g} psia",
                  f"  k = {k:g} md, kro = {kro:g}, h = {h:g} ft, mu_o = {mu:g} cp, Bo = {Bo:g}",
                  f"  re = {re_:g} ft, rw = {rw:g} ft, skin = {skin:g}, "
                  f"{'pseudo-steady' if pss else 'steady'} state"]
    else:
        lines += [f"  Reservoir pressure {pr:g} psia, bubble point {pb:g} psia, J = {J:g} STB/d/psi"]
    lines += [f"  Depth {depth:g} ft, tubing ID {D_in:g} in, roughness {rough:g}, inclination {theta:g} deg",
              f"  Wellhead pressure {ptf:g} psia, T surface/bottom {Ts:g}/{Tb:g} F, step {dz:g} ft",
              f"  GOR {GOR:g} scf/STB, WOR {WOR:g}, API {API:g}, gas gravity {gg:g}, sigma {sigma:g} dyn/cm",
              f"  PVT model: {PVT_LABELS[pvt_model]}; friction: {friction}",
              "", "RESULTS",
              f"  J = {ipr.productivity_index:.4f} STB/d/psi, IPR max rate = {qmax:,.1f} STB/d"]
    for m, r in results.items():
        if r.status == "flowing":
            lines.append(f"  {METHOD_NAMES[m]}: oil {r.q:,.1f} STB/d, liquid {r.q * (1 + WOR):,.1f} STB/d, "
                         f"gas {r.q * GOR / 1000:,.1f} Mscf/d, pwf = {r.pwf:,.1f} psia")
        else:
            lines.append(f"  {METHOD_NAMES[m]}: {r.message}")
    return "\n".join(lines)


with tab_data:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("IPR")
        ipr_df = ipr_table(ipr, 41).rename(columns={"pwf_psia": "pwf (psia)", "qo_stb_day": "qo (STB/d)"})
        st.dataframe(ipr_df.round(2), height=320)
        st.download_button("Download IPR (CSV)", ipr_df.to_csv(index=False), "ipr.csv", "text/csv")
    with c2:
        st.subheader("VLP")
        vlp_df = results[primary].table[["qo (STB/d)", "IPR pwf (psia)"]].copy()
        for m, r in results.items():
            vlp_df[f"VLP pwf {m} (psia)"] = r.table["VLP pwf (psia)"].values
        st.dataframe(vlp_df.round(2), height=320)
        st.download_button("Download VLP (CSV)", vlp_df.to_csv(index=False), "vlp.csv", "text/csv")
    st.subheader("Example-case report")
    report = build_report()
    st.code(report, language=None)
    st.download_button("Download report (TXT)", report, "nodal_case_report.txt", "text/plain")

# --------------------------------------------------------------------------
# Tab 5 - method notes
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(r"""
**Nodal analysis at the bottomhole node** (PPS Ch. 9). The operating point is where inflow (IPR) equals
outflow (VLP). Moving to higher rates, the *stable* point is where the VLP rises through the IPR; a crossing on
the falling, liquid-loading branch is unstable and is shown with a hollow marker. If the VLP lies above the
IPR everywhere, the well will not flow naturally.

**Inflow (IPR)** — PPS Ch. 2-3, 6
* Above the bubble point: $q_o = J(\bar p - p_{wf})$
* Below the bubble point (generalized Vogel, Sec. 3.5):
  $q_o = J(\bar p - p_b) + \dfrac{J p_b}{1.8}\left[1 - 0.2\dfrac{p_{wf}}{p_b} - 0.8\left(\dfrac{p_{wf}}{p_b}\right)^2\right]$
* Saturated reservoir: Vogel with $q_{max} = J\bar p/1.8$ (J = tangent PI at $\bar p$)
* Radial-flow PI with skin: $J = \dfrac{0.00708\,k\,k_{ro}\,h}{\mu_o B_o\left[\ln(r_e/r_w) - 0.75 + s\right]}$

**Vertical lift (VLP)** — PPS Sec. 7.4. Pressure traverse from the wellhead down in fixed Δz steps, with the
gradient evaluated at mid-step pressure and temperature and iterated to 0.01 psi.
* *Modified Hagedorn-Brown*: H-B holdup charts (Eqs. 7-103 to 7-107) with $H_L \ge \lambda_L$; Griffith bubble-flow
  correlation when $\lambda_g < L_B$, with friction on the in-situ liquid velocity (Eqs. 7-111 to 7-116).
* *Beggs-Brill* with the Payne et al. upward-flow holdup correction (0.924 factor), inclination correction (Table 7-2).

**Fluid properties**
* *Textbook correlations*: Vasquez-Beggs $R_s$, $B_o$ (Eqs. 3-8 to 3-15); oil density (3-17, 3-18); Beggs-Robinson viscosity
  (3-19 to 3-27); Standing pseudocriticals (4-8, 4-9); Dranchuk-Abou-Kassem Z (4-5, 4-6); Lee-Gonzalez-Eakin gas viscosity
  (4-23 to 4-26). The produced-fluid bubble point is where $R_s$ = producing GOR.
* *Appendix-B curves*: the curve fits used in Homework 5 / Exam 1 (one fluid, kept for comparison).

**Validation** — 42 automated tests (`python -m pytest`) reproduce textbook Examples 3-3, 7-8 and 7-9, check the gas
correlations, and match the Homework 5 / Exam 1 notebook results.

*Reference: Economides, Hill, Ehlig-Economides & Zhu, Petroleum Production Systems, 2nd ed.*
""")
