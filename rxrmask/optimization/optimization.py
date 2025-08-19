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
    scalar_cost_by_layers,
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
    init="latinhypercube",
    atol=0,
    updating="immediate",
):
    for rscan in ref_scans:
        rscan.R = ctx.transform.apply_R(rscan.R)
    for escan in en_scans:
        escan.R = ctx.transform.apply_R(escan.R)

    bounds = [(float(p.lower), float(p.upper)) for p in params]

    print("Data ready")
    ret = optimize.differential_evolution(
        scalar_cost,
        bounds,
        args=[params, ctx, ref_scans, en_scans],
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        atol=atol,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        init=init,
        updating=updating,
        disp=True,
        callback=None,
    )

    return ret.x, ret.fun





def fit_differential_evolution_by_layers(
    layers_to_update: List[bool],
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
    init="latinhypercube",
    atol=0,
    updating="immediate",
):
    for rscan in ref_scans:
        rscan.R = ctx.transform.apply_R(rscan.R)
    for escan in en_scans:
        escan.R = ctx.transform.apply_R(escan.R)

    x0 = []
    for el in ctx.structure.atoms_layers:
        x0.extend([el.molar_density[i] for i, update in enumerate(layers_to_update) if update])


    num_true = sum(layers_to_update)

    bounds = [(float(0), float(0.09)) for p in layers_to_update if p] * len(ctx.structure.atoms_layers)

    print("len of bounds:", len(bounds))
    print("len of x0:", len(x0))

    ret = optimize.differential_evolution(
        scalar_cost_by_layers,
        bounds,
        args=[layers_to_update, ctx, ref_scans, en_scans, num_true],
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        atol=atol,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        init=init,
        updating=updating,
        disp=True,
        callback=None,
        x0=x0,
    )

    return ret.x, ret.fun






