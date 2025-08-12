from typing import List
import numpy as np
from scipy import optimize

from rxrmask.core import (
    EnergyScan,
    ReflectivityScan,
    Parameter,
)
from .fitting import (
    scalar_cost,
    vector_residuals,
    FitContext,
)


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
        x0=np.array([p.get() for p in params], dtype=float),
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
