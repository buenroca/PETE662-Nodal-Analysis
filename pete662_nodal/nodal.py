"""Nodal analysis at the bottomhole node: IPR (inflow) vs. VLP (outflow).
PPS Ch. 9, Sec. 9.2 (combination of IPR and VPR).

Stability rule: moving to higher rate, the STABLE operating point is where
the VLP rises through the IPR (VLP - IPR goes from negative to positive).
A crossing in the other direction (on the falling, liquid-loading branch of
the VLP) is unstable and is reported but not used.  If the VLP lies above
the IPR everywhere, the well will not flow naturally.
"""
from dataclasses import dataclass, field, fields, replace
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .inputs import WellboreCase
from .pvt import make_pvt
from .ipr import IPRInputs, ReservoirInputs, oil_rate, pwf_from_rate
from .traverse import pressure_traverse


@dataclass
class NodalResult:
    status: str                       # "flowing" | "no natural flow" | "no crossing in range"
    q: Optional[float]                # STB/d at the stable operating point
    pwf: Optional[float]              # psia
    message: str
    table: pd.DataFrame               # q, VLP pwf, IPR pwf, VLP - IPR on the search grid
    unstable_points: List[Tuple[float, float]] = field(default_factory=list)

    def __repr__(self):
        if self.status == "flowing":
            return f"NodalResult(flowing: q = {self.q:.1f} STB/d, pwf = {self.pwf:.1f} psia)"
        return f"NodalResult({self.status}: {self.message})"


def operating_point(inflow, case: WellboreCase, method="HB", n_points=30, pvt=None) -> NodalResult:
    """Find the stable IPR/VLP intersection.

    inflow : IPRInputs or ReservoirInputs
    """
    ipr = inflow.to_ipr()
    pvt = pvt or make_pvt(case)
    qmax = oil_rate(0.0, ipr)
    q_grid = np.linspace(0.02 * qmax, 0.995 * qmax, n_points)

    vlp = np.array([pressure_traverse(q, case, method, pvt) for q in q_grid])
    iprp = np.array([pwf_from_rate(q, ipr) for q in q_grid])
    d = vlp - iprp
    table = pd.DataFrame({"qo (STB/d)": q_grid, "VLP pwf (psia)": vlp,
                          "IPR pwf (psia)": iprp, "VLP - IPR (psi)": d})

    def mismatch(q):
        return pressure_traverse(q, case, method, pvt) - pwf_from_rate(q, ipr)

    stable, unstable = [], []
    for i in range(len(q_grid) - 1):
        if d[i] < 0 <= d[i + 1] or d[i] <= 0 < d[i + 1]:
            stable.append((q_grid[i], q_grid[i + 1]))
        elif d[i] > 0 >= d[i + 1] or d[i] >= 0 > d[i + 1]:
            unstable.append((q_grid[i], q_grid[i + 1]))

    unstable_pts = []
    for a, b in unstable:
        q = brentq(mismatch, a, b, xtol=0.01)
        unstable_pts.append((q, pwf_from_rate(q, ipr)))

    if stable:
        a, b = stable[-1]                         # highest-rate stable crossing
        q = brentq(mismatch, a, b, xtol=0.01)
        return NodalResult("flowing", q, pwf_from_rate(q, ipr),
                           "Stable operating point found.", table, unstable_pts)
    if np.all(d > 0):
        return NodalResult("no natural flow", None, None,
                           "VLP is above the IPR at every rate: the well will not flow "
                           "naturally at these conditions (consider artificial lift, "
                           "lower wellhead pressure or different tubing).", table, unstable_pts)
    return NodalResult("no crossing in range", None, None,
                       "No stable crossing found between 2% and 99.5% of the IPR maximum rate.",
                       table, unstable_pts)


def _replace_any(inflow, case, name, value):
    if name in {f.name for f in fields(case)}:
        return inflow, replace(case, **{name: value})
    if name in {f.name for f in fields(inflow)}:
        return replace(inflow, **{name: value}), case
    raise ValueError(f"'{name}' is not a WellboreCase or {type(inflow).__name__} field.")


def run_sensitivity(inflow, case: WellboreCase, name: str, values, method="HB", n_points=30):
    """Vary one input (any WellboreCase / IPR / ReservoirInputs field), hold the rest.

    Examples: name="D_in", "ptf", "GOR", "WOR"  (wellbore)
              name="skin", "k_md", "reservoir_pressure"  (ReservoirInputs)
              name="productivity_index"  (IPRInputs)
    Returns (summary DataFrame, list of NodalResult).
    """
    rows, results = [], []
    for v in values:
        inf_v, case_v = _replace_any(inflow, case, name, v)
        r = operating_point(inf_v, case_v, method, n_points)
        results.append(r)
        rows.append({name: v, "status": r.status,
                     "q (STB/d)": r.q, "pwf (psia)": r.pwf,
                     "J (STB/d/psi)": inf_v.to_ipr().productivity_index})
    return pd.DataFrame(rows), results
