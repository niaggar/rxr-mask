from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Literal
import numpy as np
from scipy import optimize

# ---- Types from your lib we import at call sites (no runtime import here) ----
from rxrmask.backends import ReflectivityBackend
from rxrmask.core import Structure, ParametersContainer, ReflectivityData, EnergyScanData

RScale = Literal["x", "log10", "ln", "qz4"]
Objective = Literal["chi2", "l1", "l2", "atan"]

# -------------------- Scan descriptors (measured data) --------------------

@dataclass
class ReflectivityScan:
    name: str
    energy_eV: float
    pol: Literal["s", "p"]  # which polarization to use from the simulation
    qz: np.ndarray          # measured qz grid
    R: np.ndarray           # measured reflectivity (same length as qz)
    Rsmooth: Optional[np.ndarray] = None
    bounds: List[Tuple[float, float]] = None
    weights: List[float] = None
    # per-scan knobs (can be tied to ParametersContainer if you prefer)
    background_shift: float = 0.0
    scale_factor: float = 1.0

@dataclass
class EnergyScan:
    name: str
    theta_deg: float
    pol: Literal["s", "p"]
    E_eV: np.ndarray        # measured energy grid
    R: np.ndarray           # measured reflectivity (same length as E_eV)
    Rsmooth: Optional[np.ndarray] = None
    bounds: List[Tuple[float, float]] = None
    weights: List[float] = None
    qz_const: Optional[float] = None  # if you know qz (your EnergyScanData carries it)
    background_shift: float = 0.0
    scale_factor: float = 1.0

# -------------------- Transforms & regularization --------------------

@dataclass
class FitTransform:
    r_scale: RScale = "x"
    eps: float = 1e-30  # to avoid log(0)

    def apply_R(self, R: np.ndarray, *, qz: Optional[np.ndarray] = None,
                theta_deg: Optional[float] = None,
                E_eV: Optional[np.ndarray] = None,
                qz_const: Optional[float] = None) -> np.ndarray:
        if self.r_scale == "x":
            return R
        if self.r_scale == "log10":
            return np.log10(np.clip(R, self.eps, None))
        if self.r_scale == "ln":
            return np.log(np.clip(R, self.eps, None))
        if self.r_scale == "qz4":
            if qz is None:
                if qz_const is not None and E_eV is not None:
                    # Energy scan with known constant qz from your EnergyScanData
                    qz = np.full_like(E_eV, float(qz_const))
                elif theta_deg is not None and E_eV is not None:
                    # fallback: qz from theta, E (keV->Å^-1 factor from legacy code)
                    qz = np.sin(np.deg2rad(theta_deg)) * (E_eV * 0.001013546143)
            if qz is None:
                raise ValueError("qz^4 scaling requires qz or (theta,E) or qz_const.")
            return R * (qz**4)
        raise ValueError(self.r_scale)

@dataclass
class TVRegularizer:
    weight: float = 0.0
    def penalty(self, Rs: np.ndarray, Rt: np.ndarray) -> float:
        if self.weight == 0.0:
            return 0.0
        if len(Rs) < 2 or len(Rt) < 2:
            return 0.0
        tv_s = np.abs(np.diff(Rs)).mean()
        tv_t = np.abs(np.diff(Rt)).mean()
        return self.weight * abs(tv_s - tv_t)

# -------------------- Param binding to your ParametersContainer --------------------

@dataclass
class ParamSpec:
    name: str
    get: Callable[[], float]
    set: Callable[[float], None]
    lower: float
    upper: float

def params_from_container(container, structure,
                          post_set: Optional[Callable[[], None]] = None) -> List[ParamSpec]:
    """
    Build ParamSpec list from your ParametersContainer.
    - Uses Parameter.fit flag to select.
    - Uses min_value/max_value when available (fallbacks if None).
    - Calls post_set() after *every* parameter write (to keep layers/derived in sync).
    """
    specs: List[ParamSpec] = []
    for p in container.parameters:
        if not getattr(p, "fit", False):
            continue
        lo = p.min_value if p.min_value is not None else -np.inf
        hi = p.max_value if p.max_value is not None else  np.inf

        def make_setter(prm):
            def setter(v: float):
                prm.set(float(v))
                # keep derived + discretization in sync
                container.update_derived()
                # if thickness/density/roughness changed, refresh layers
                # assumes element_data already built by create_layers():
                if hasattr(structure, "update_layers"):
                    structure.update_layers()
                if post_set is not None:
                    post_set()
            return setter

        specs.append(ParamSpec(
            name=p.name,
            get=lambda prm=p: float(prm.get()),
            set=make_setter(p),
            lower=float(lo),
            upper=float(hi),
        ))
    return specs

# -------------------- Utility --------------------

def _mask_by_bounds(x: np.ndarray, bounds: List[Tuple[float,float]]) -> np.ndarray:
    if not bounds:
        return np.ones_like(x, dtype=bool)
    mask = np.zeros_like(x, dtype=bool)
    for lo, hi in bounds:
        mask |= (x >= lo) & (x < hi)
    return mask

def _objective(diff: np.ndarray, ysim: np.ndarray, kind: Objective) -> float:
    if diff.size == 0:
        return 0.0
    if kind == "chi2":
        denom = np.clip(np.abs(ysim), 1e-20, None)
        return float(np.sum((diff**2) / denom))
    if kind == "l1":
        return float(np.sum(np.abs(diff)))
    if kind == "l2":
        return float(np.sum(diff**2))
    if kind == "atan":
        return float(np.sum(np.arctan(diff**2)))
    raise ValueError(kind)

# -------------------- Cost/residuals (backend-agnostic) --------------------

@dataclass
class FitContext:
    backend: ReflectivityBackend
    structure: "Structure"
    rtx_table: Dict[str, np.ndarray] | None  # e.g., orbital FF tables if you have them; else None
    transform: FitTransform
    tv: TVRegularizer
    objective: Objective
    s_min: float = 0.1  # not used here; your backend already encapsulates precision/ALS
    # You can add hooks here (e.g., callback before/after sim)

def scalar_cost(x: np.ndarray, params: List[ParamSpec],
                ctx: FitContext,
                ref_scans: List[ReflectivityScan],
                en_scans: List[EnergyScan]) -> float:
    # apply params
    for val, spec in zip(x, params):
        spec.set(float(val))

    total = 0.0

    # Reflectivity scans (fixed E, varying qz)
    for sc in ref_scans:
        # simulate
        sim: "ReflectivityData" = ctx.backend.compute_reflectivity(ctx.structure, sc.qz, sc.energy_eV)
        Rsim = sim.R_s if sc.pol == "s" else sim.R_p
        # shift/scale, clip
        Rsim = np.clip(sc.background_shift + sc.scale_factor * Rsim, 0.0, None)

        # bounds mask
        mask = _mask_by_bounds(sc.qz, sc.bounds or [])
        qz_use = sc.qz[mask]
        Rdat = sc.R[mask]
        Rsmooth = (sc.Rsmooth if sc.Rsmooth is not None else sc.R)[mask]

        Rt = ctx.transform.apply_R(Rsim[mask], qz=qz_use)
        Rd = ctx.transform.apply_R(Rdat, qz=qz_use)
        Rs = ctx.transform.apply_R(Rsmooth, qz=qz_use)

        total += _objective(Rd - Rt, Rt, ctx.objective)
        total += ctx.tv.penalty(Rs, Rt)

    # Energy scans (fixed theta, varying E)
    for sc in en_scans:
        sim: "EnergyScanData" = ctx.backend.compute_energy_scan(ctx.structure, sc.E_eV, sc.theta_deg)
        Rsim = sim.R_s if sc.pol == "s" else sim.R_p
        Rsim = np.clip(sc.background_shift + sc.scale_factor * Rsim, 0.0, None)

        mask = _mask_by_bounds(sc.E_eV, sc.bounds or [])
        E_use = sc.E_eV[mask]
        Rdat = sc.R[mask]
        Rsmooth = (sc.Rsmooth if sc.Rsmooth is not None else sc.R)[mask]

        Rt = ctx.transform.apply_R(Rsim[mask], theta_deg=sc.theta_deg, E_eV=E_use, qz_const=(sim.qz if hasattr(sim, "qz") else sc.qz_const))
        Rd = ctx.transform.apply_R(Rdat, theta_deg=sc.theta_deg, E_eV=E_use, qz_const=(sim.qz if hasattr(sim, "qz") else sc.qz_const))
        Rs = ctx.transform.apply_R(Rsmooth, theta_deg=sc.theta_deg, E_eV=E_use, qz_const=(sim.qz if hasattr(sim, "qz") else sc.qz_const))

        total += _objective(Rd - Rt, Rt, ctx.objective)
        total += ctx.tv.penalty(Rs, Rt)

    return total

def vector_residuals(x: np.ndarray, params: List[ParamSpec],
                     ctx: FitContext,
                     ref_scans: List[ReflectivityScan],
                     en_scans: List[EnergyScan]) -> np.ndarray:
    for val, spec in zip(x, params):
        spec.set(float(val))

    res = []

    for sc in ref_scans:
        sim = ctx.backend.compute_reflectivity(ctx.structure, sc.qz, sc.energy_eV)
        Rsim = sim.R_s if sc.pol == "s" else sim.R_p
        Rsim = np.clip(sc.background_shift + sc.scale_factor * Rsim, 0.0, None)

        mask = _mask_by_bounds(sc.qz, sc.bounds or [])
        qz_use = sc.qz[mask]
        Rdat = sc.R[mask]

        Rt = ctx.transform.apply_R(Rsim[mask], qz=qz_use)
        Rd = ctx.transform.apply_R(Rdat, qz=qz_use)
        res.append(Rd - Rt)

    for sc in en_scans:
        sim = ctx.backend.compute_energy_scan(ctx.structure, sc.E_eV, sc.theta_deg)
        Rsim = sim.R_s if sc.pol == "s" else sim.R_p
        Rsim = np.clip(sc.background_shift + sc.scale_factor * Rsim, 0.0, None)

        mask = _mask_by_bounds(sc.E_eV, sc.bounds or [])
        E_use = sc.E_eV[mask]
        Rdat = sc.R[mask]

        Rt = ctx.transform.apply_R(Rsim[mask], theta_deg=sc.theta_deg, E_eV=E_use, qz_const=(sim.qz if hasattr(sim, "qz") else sc.qz_const))
        Rd = ctx.transform.apply_R(Rdat, theta_deg=sc.theta_deg, E_eV=E_use, qz_const=(sim.qz if hasattr(sim, "qz") else sc.qz_const))
        res.append(Rd - Rt)

    return np.concatenate(res) if res else np.zeros(0)

# -------------------- Optimizer wrappers --------------------

def fit_differential_evolution(params: List[ParamSpec], ctx: FitContext,
                               ref_scans: List[ReflectivityScan],
                               en_scans: List[EnergyScan],
                               *, strategy="best1bin", maxiter=200, popsize=20, tol=1e-6,
                               mutation=(0.5, 1.0), recombination=0.7,
                               polish=True, seed=None, updating="deferred"):
    bounds = [(p.lower, p.upper) for p in params]
    ret = optimize.differential_evolution(
        lambda x: scalar_cost(x, params, ctx, ref_scans, en_scans),
        bounds=bounds, strategy=strategy, maxiter=maxiter, popsize=popsize,
        tol=tol, mutation=mutation, recombination=recombination,
        polish=polish, seed=seed, updating=updating, disp=False
    )
    return ret.x, ret.fun

def fit_least_squares(x0: np.ndarray, params: List[ParamSpec], ctx: FitContext,
                      ref_scans: List[ReflectivityScan],
                      en_scans: List[EnergyScan],
                      *, method="trf", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=None):
    lb = np.array([p.lower for p in params], dtype=float)
    ub = np.array([p.upper for p in params], dtype=float)
    res = optimize.least_squares(
        lambda x: vector_residuals(x, params, ctx, ref_scans, en_scans),
        x0=np.asarray(x0, dtype=float), bounds=(lb, ub),
        method=method, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev
    )
    cov = None
    if res.jac is not None and res.jac.size:
        JTJ = res.jac.T @ res.jac
        try:
            cov = np.linalg.inv(JTJ)
        except np.linalg.LinAlgError:
            pass
    return res.x, float(res.cost), cov
