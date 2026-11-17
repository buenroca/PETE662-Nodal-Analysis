"""Wellbore / fluid inputs for the VLP side of the nodal analysis.

Field names deliberately match the global variables used in the
Homework 5 / Exam 1 notebooks (depth, ptf, D_in, GOR, ...), so the
notebook code maps across one-to-one.  The difference is that every
calculation now receives a ``WellboreCase`` instead of reading globals,
which is what makes sensitivity runs possible:

    from dataclasses import replace
    bigger_tubing = replace(case, D_in=2.992)
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WellboreCase:
    # ---------------- WELL ----------------
    depth: float              # ft, measured depth of the traverse (wellhead -> node)
    ptf: float                # psia, flowing tubing-head pressure
    T_surface: float          # deg F
    T_bottom: float           # deg F
    # ---------------- TUBING --------------
    D_in: float               # in, tubing inside diameter
    rel_roughness: float = 0.0006
    theta_deg: float = 90.0   # inclination from horizontal; 90 = vertical upward
    dz: float = 200.0         # ft, traverse increment
    friction_method: str = "haaland"   # "haaland" (notebooks) or "chen" (PPS Eq. 7-35)
    pvt_model: str = "appendix_b"      # "appendix_b" (HW5 curves) or "correlations" (Ch. 3-4)
    # ---------------- PRODUCTION ----------
    GOR: float = 400.0        # scf/STB, producing gas-oil ratio
    WOR: float = 0.0          # STB water / STB oil
    # ---------------- FLUID ---------------
    API: float = 32.0
    gamma_g: float = 0.71
    gamma_w: float = 1.0
    pb: float = 4336.0        # psia
    Rsb: float = 800.0        # scf/STB
    sigma_L: float = 30.0     # dyn/cm
    mu_g: float = 0.015       # cp
    mu_w: float = 0.7         # cp

    def __post_init__(self):
        positive = dict(depth=self.depth, ptf=self.ptf, D_in=self.D_in, dz=self.dz,
                        pb=self.pb, Rsb=self.Rsb, sigma_L=self.sigma_L, mu_g=self.mu_g,
                        gamma_g=self.gamma_g, API=self.API)
        for name, val in positive.items():
            if not (math.isfinite(val) and val > 0):
                raise ValueError(f"{name} must be a positive number (got {val}).")
        if self.GOR < 0 or self.WOR < 0:
            raise ValueError("GOR and WOR cannot be negative.")
        if not 0 < self.theta_deg <= 90:
            raise ValueError("theta_deg must be in (0, 90] for upward flow.")

    # Derived geometry ------------------------------------------------
    @property
    def D(self) -> float:
        """Tubing ID, ft."""
        return self.D_in / 12.0

    @property
    def A(self) -> float:
        """Tubing flow area, ft^2."""
        return math.pi * self.D ** 2 / 4.0

    def T_at_depth(self, z: float) -> float:
        """Linear temperature profile, deg F."""
        return self.T_surface + (self.T_bottom - self.T_surface) * z / self.depth
