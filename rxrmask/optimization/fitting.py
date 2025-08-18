from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal
import numpy as np

from rxrmask.backends import ReflectivityBackend
from rxrmask.core import (
    Structure,
    EnergyScan,
    ReflectivityScan,
    Parameter,
)

RScale = Literal["x", "log10", "ln", "qz4"]
Objective = Literal["chi2", "l1", "l2", "atan"]


@dataclass
class FitTransform:
    sim_scale: Parameter
    sim_offset: Parameter
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

    def apply_correction(self, R: np.ndarray) -> np.ndarray:
        return self.sim_scale.get() * R + self.sim_offset.get()


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


@dataclass
class FitContext:
    backend: ReflectivityBackend
    structure: Structure
    transform: FitTransform
    tv: TVRegularizer
    objective: Objective


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
        return float(np.sum((diff**2) / np.abs(ysim)))
    if kind == "l1":
        return float(np.sum(np.abs(diff)))
    if kind == "l2":
        return float(np.sum(diff**2))
    if kind == "atan":
        return float(np.sum(np.arctan(diff**2)))


def scalar_cost(
    x: np.ndarray,
    params: List[Parameter],
    ctx: FitContext,
    ref_scans: List[ReflectivityScan],
    en_scans: List[EnergyScan],
) -> float:
    for val, spec in zip(x, params):
        spec.set(float(val))
    ctx.structure.update_layers()

    total = 0.0
    # Reflectivity scans (fixed E, varying qz)
    for sc in ref_scans:
        sim_theta = ctx.backend.compute_reflectivity(ctx.structure, sc.qz, sc.energy_eV)
        Rsim = sim_theta.R_s if sc.pol == "s" else sim_theta.R_p
        # mask = _mask_by_bounds(sc.qz, sc.bounds or [])
        Rdat = sc.R
        # Rsmooth = sc.R[mask]

        Rsim = ctx.transform.apply_correction(Rsim)
        Rt = ctx.transform.apply_R(Rsim)
        # Rs = ctx.transform.apply_R(Rsmooth)

        total += _objective(Rdat - Rt, Rt, ctx.objective)
        # total += ctx.tv.penalty(Rs, Rt)

    # Energy scans (fixed theta, varying E)
    for sc in en_scans:
        sim_energy = ctx.backend.compute_energy_scan(ctx.structure, sc.E_eV, sc.theta_deg)
        Rsim = sim_energy.R_s if sc.pol == "s" else sim_energy.R_p
        # mask = _mask_by_bounds(sc.E_eV, sc.bounds or [])
        Rdat = sc.R
        # Rsmooth = sc.R[mask]

        Rsim = ctx.transform.apply_correction(Rsim)
        Rt = ctx.transform.apply_R(Rsim)
        # Rs = ctx.transform.apply_R(Rsmooth)

        total += _objective(Rdat - Rt, Rt, ctx.objective)
        # total += ctx.tv.penalty(Rs, Rt)

    print(f"Total cost: {total:.6f}")
    return total
