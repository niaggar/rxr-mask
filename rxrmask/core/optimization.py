from typing import Callable, Literal
from scipy.optimize import minimize, differential_evolution

from rxrmask.core.parameter import ParametersContainer
from rxrmask.core.structure import Structure

def optimize_structure_parameters(
    structure: Structure,
    container: ParametersContainer,
    model_loss: Callable[[Structure], float],
    method: Literal["nelder", "l-bfgs", "de"] = "nelder",
    bounds: list[tuple[float, float]] | None = None,
    verbose: bool = True
):
    def wrapped_loss(x):
        container.set_fit_vector(x)
        structure.update_layers()
        return model_loss(structure)

    x0 = container.get_fit_vector()

    if method == "nelder":
        result = minimize(wrapped_loss, x0, method="Nelder-Mead")
    elif method == "l-bfgs":
        result = minimize(wrapped_loss, x0, method="L-BFGS-B")
    elif method == "de":
        if bounds is None:
            raise ValueError("Bounds must be provided for differential evolution method.")
        result = differential_evolution(wrapped_loss, bounds)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    container.set_fit_vector(result.x)
    structure.update_layers()

    if verbose:
        print("\n--- Optimization complete ---")
        print("Success:", result.success)
        print("Message:", result.message)
        print("Final loss:", result.fun)
        print("------------------------------\n")

    return {
        "x": result.x,
        "loss": result.fun
    }



