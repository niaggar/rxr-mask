from scipy import optimize


class Optimizer:
    """Base class for optimization algorithms."""

    def minimize(self, x0, loss_fn, grad_fn=None, bounds=None):
        """Minimize the loss function."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class NelderMeadOptimizer(Optimizer):
    def minimize(self, x0, loss_fn, grad_fn=None, bounds=None):
        result = optimize.minimize(loss_fn, x0, method="Nelder-Mead")
        return result


class LBFGSOptimizer(Optimizer):
    def minimize(self, x0, loss_fn, grad_fn=None, bounds=None):
        result = optimize.minimize(loss_fn, x0, method="L-BFGS-B", jac=grad_fn, bounds=bounds)
        return result


class DifferentialEvolutionOptimizer(Optimizer):
    def minimize(self, x0, loss_fn, grad_fn=None, bounds=None):
        result = optimize.differential_evolution(loss_fn, bounds)
        return result


class LeastSquaresOptimizer(Optimizer):
    def minimize(self, x0, loss_fn, grad_fn=None, bounds=None):
        result = optimize.least_squares(loss_fn, x0)
        return result
