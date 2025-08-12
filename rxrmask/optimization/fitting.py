from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal
import numpy as np
from scipy import optimize

from rxrmask.backends import ReflectivityBackend
from rxrmask.core import (
    Structure,
    ParametersContainer,
    SimReflectivityData,
    SimEnergyScanData,
    EnergyScan,
    ReflectivityScan,
    Parameter,
)

RScale = Literal["x", "log10", "ln", "qz4"]
Objective = Literal["chi2", "l1", "l2", "atan"]


# -------------------- Transforms & regularization --------------------


@dataclass
class FitTransform:
    r_scale: RScale = "x"

    def apply_R(
        self,
        R: np.ndarray,
    ) -> np.ndarray:
        if self.r_scale == "x":
            return R
        if self.r_scale == "log10":
            return np.log10(R)
        if self.r_scale == "ln":
            return np.log(R)
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


# -------------------- Utility --------------------


def _mask_by_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
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


# -------------------- Cost/residuals (backend-agnostic) --------------------


@dataclass
class FitContext:
    backend: ReflectivityBackend
    structure: Structure
    rtx_table: Dict[str, np.ndarray] | None  # e.g., orbital FF tables if you have them; else None
    transform: FitTransform
    tv: TVRegularizer
    objective: Objective


def scalar_cost(
    x: np.ndarray,
    params: List[Parameter],
    ctx: FitContext,
    ref_scans: List[ReflectivityScan],
    en_scans: List[EnergyScan],
) -> float:
    # apply params
    for val, spec in zip(x, params):
        spec.set(float(val))

    total = 0.0

    # Reflectivity scans (fixed E, varying qz)
    for sc in ref_scans:
        sim_theta = ctx.backend.compute_reflectivity(ctx.structure, sc.qz, sc.energy_eV)
        Rsim = sim_theta.R_s if sc.pol == "s" else sim_theta.R_p
        Rsim = np.clip(sc.background_shift + sc.scale_factor * Rsim, 0.0, None)

        mask = _mask_by_bounds(sc.qz, sc.bounds or [])
        Rdat = sc.R[mask]
        Rsmooth = sc.R[mask]

        Rt = ctx.transform.apply_R(Rsim[mask])
        Rd = ctx.transform.apply_R(Rdat)
        Rs = ctx.transform.apply_R(Rsmooth)

        total += _objective(Rd - Rt, Rt, ctx.objective)
        total += ctx.tv.penalty(Rs, Rt)

    # Energy scans (fixed theta, varying E)
    for sc in en_scans:
        sim_energy = ctx.backend.compute_energy_scan(ctx.structure, sc.E_eV, sc.theta_deg)
        Rsim = sim_energy.R_s if sc.pol == "s" else sim_energy.R_p
        Rsim = np.clip(sc.background_shift + sc.scale_factor * Rsim, 0.0, None)

        mask = _mask_by_bounds(sc.E_eV, sc.bounds or [])
        Rdat = sc.R[mask]
        Rsmooth = sc.R[mask]

        Rt = ctx.transform.apply_R(Rsim[mask])
        Rd = ctx.transform.apply_R(Rdat)
        Rs = ctx.transform.apply_R(Rsmooth)

        total += _objective(Rd - Rt, Rt, ctx.objective)
        total += ctx.tv.penalty(Rs, Rt)

    return total


def vector_residuals(
    x: np.ndarray,
    params: List[Parameter],
    ctx: FitContext,
    ref_scans: List[ReflectivityScan],
    en_scans: List[EnergyScan],
) -> np.ndarray:
    for val, spec in zip(x, params):
        spec.set(float(val))

    res = []

    for sc in ref_scans:
        sim = ctx.backend.compute_reflectivity(ctx.structure, sc.qz, sc.energy_eV)
        Rsim = sim.R_s if sc.pol == "s" else sim.R_p
        Rsim = np.clip(sc.background_shift + sc.scale_factor * Rsim, 0.0, None)

        mask = _mask_by_bounds(sc.qz, sc.bounds or [])
        Rdat = sc.R[mask]

        Rt = ctx.transform.apply_R(Rsim[mask])
        Rd = ctx.transform.apply_R(Rdat)
        res.append(Rd - Rt)

    for sc in en_scans:
        sim = ctx.backend.compute_energy_scan(ctx.structure, sc.E_eV, sc.theta_deg)
        Rsim = sim.R_s if sc.pol == "s" else sim.R_p
        Rsim = np.clip(sc.background_shift + sc.scale_factor * Rsim, 0.0, None)

        mask = _mask_by_bounds(sc.E_eV, sc.bounds or [])
        Rdat = sc.R[mask]

        Rt = ctx.transform.apply_R(Rsim[mask])
        Rd = ctx.transform.apply_R(Rdat)
        res.append(Rd - Rt)

    return np.concatenate(res) if res else np.zeros(0)


# -------------------- Optimizer wrappers --------------------


def fit_differential_evolution(
    params: List[Parameter],
    ctx: FitContext,
    ref_scans: List[ReflectivityScan],
    en_scans: List[EnergyScan],
    strategy="best1bin",
    maxiter=200,
    popsize=20,
    tol=1e-6,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=True,
    updating="deferred",
):
    bounds = [(p.lower, p.upper) for p in params]
    ret = optimize.differential_evolution(
        lambda x: scalar_cost(x, params, ctx, ref_scans, en_scans),
        bounds=bounds,
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        updating=updating,
        disp=False,
    )
    return ret.x, ret.fun


def fit_least_squares(
    x0: np.ndarray,
    params: List[Parameter],
    ctx: FitContext,
    ref_scans: List[ReflectivityScan],
    en_scans: List[EnergyScan],
    *,
    method="trf",
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    max_nfev=None,
):
    lb = np.array([p.lower for p in params], dtype=float)
    ub = np.array([p.upper for p in params], dtype=float)
    res = optimize.least_squares(
        lambda x: vector_residuals(x, params, ctx, ref_scans, en_scans),
        x0=np.asarray(x0, dtype=float),
        bounds=(lb, ub),
        method=method,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
    )

    print(f"Least squares optimization result: {res.message}, cost: {res.cost}")
    print(res)

    return res.x, float(res.cost)
