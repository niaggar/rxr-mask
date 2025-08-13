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
    scalar_cost_layers,
    vector_residuals,
    FitContext,
)


def fit_differential_evolution(
    x0: np.ndarray,
    params: List[Parameter],
    ctx: FitContext,
    ref_scans: List[ReflectivityScan],
    en_scans: List[EnergyScan],
    strategy="best1bin",
    maxiter=20,
    popsize=2,
    tol=1,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=False,
    updating="immediate",
):
    for rscan in ref_scans:
        rscan.R = ctx.transform.apply_R(rscan.R)
    for escan in en_scans:
        escan.R = ctx.transform.apply_R(escan.R)
    
    bounds = [(p.lower, p.upper) for p in params]
    ret = optimize.differential_evolution(
        lambda x: scalar_cost(x, params, ctx, ref_scans, en_scans),
        x0=x0,
        init="sobol",
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
    
    # ret = optimize.differential_evolution(
    #     _objective,
    #     bounds=bounds,
    #     args=(params, ctx, ref_scans, en_scans),  # <- pasa args aquí
    #     x0=x0,
    #     strategy=strategy,
    #     maxiter=maxiter,
    #     popsize=popsize,
    #     tol=tol,
    #     mutation=mutation,
    #     recombination=recombination,
    #     polish=polish,
    #     updating="deferred",  # bueno para paralelizar
    #     disp=False,
    #     workers=-1,           # ahora sí usa procesos
    #     init="sobol",         # si usas x0, init se ignora; si no, deja 'sobol'
    #     callback=callback,    # si lo usas
    # )

    print(f"Differential evolution optimization result: {ret.message}, cost: {ret.fun}")
    return ret.x, ret.fun


def fit_least_squares(
    x0: np.ndarray,
    params: List[Parameter],
    ctx: FitContext,
    ref_scans: List[ReflectivityScan],
    en_scans: List[EnergyScan],
    method="trf",
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    max_nfev=None,
):
    for rscan in ref_scans:
        rscan.R = ctx.transform.apply_R(rscan.R)
    for escan in en_scans:
        escan.R = ctx.transform.apply_R(escan.R)

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

    return res.x, float(res.cost)



def fit_differential_evolution_layers(
    ctx: FitContext,
    ref_scans: List[ReflectivityScan],
    en_scans: List[EnergyScan],
    strategy="best1bin",
    maxiter=20,
    popsize=15,
    tol=1e-0,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=False,
    updating="immediate",
):
    for rscan in ref_scans:
        rscan.R = ctx.transform.apply_R(rscan.R)
    for escan in en_scans:
        escan.R = ctx.transform.apply_R(escan.R)

    x0 = []
    for layer in ctx.structure.atoms_layers:
        x0 = np.append(x0, layer.molar_density)

    bounds = [(0.0, 0.5) for _ in x0]

    ret = optimize.differential_evolution(
        lambda x: scalar_cost_layers(x, ctx, ref_scans, en_scans),
        x0=x0,
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

    print(f"Differential evolution optimization result: {ret.message}, cost: {ret.fun}")
    return ret.x, ret.fun