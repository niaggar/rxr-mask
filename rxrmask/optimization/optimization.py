from typing import List
import numpy as np
from scipy import optimize
from scipy.optimize import Bounds

from rxrmask.core import (
    EnergyScan,
    ReflectivityScan,
    Parameter,
)
from .fitting import (
    scalar_cost,
    FitContext,
)


def fit_differential_evolution(
    params: List[Parameter],
    ctx: FitContext,
    ref_scans: List[ReflectivityScan],
    en_scans: List[EnergyScan],
    strategy="currenttobest1bin",
    maxiter=50,
    popsize=15,
    tol=1e-06,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=False,
    updating="immediate",
):
    for rscan in ref_scans:
        rscan.R = ctx.transform.apply_R(rscan.R)
    for escan in en_scans:
        escan.R = ctx.transform.apply_R(escan.R)

    bounds = [(float(p.lower), float(p.upper)) for p in params]
    ret = optimize.differential_evolution(
        scalar_cost,
        bounds,
        args=[params, ctx, ref_scans, en_scans],
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        atol=0,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        init="latinhypercube",
        updating=updating,
        disp=True,
        callback=None,
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
