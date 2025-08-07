from scipy.optimize import minimize, differential_evolution



class Optimizer:
    def minimize(self, x0, loss_fn, grad_fn=None, bounds=None):
        raise NotImplementedError("This method should be overridden by subclasses.")


class NelderMeadOptimizer(Optimizer):
    def minimize(self, x0, loss_fn, grad_fn=None, bounds=None):
        result = minimize(loss_fn, x0, method="Nelder-Mead")
        return result

class LBFGSOptimizer(Optimizer):
    def minimize(self, x0, loss_fn, grad_fn=None, bounds=None):
        result = minimize(loss_fn, x0, method="L-BFGS-B", jac=grad_fn, bounds=bounds)
        return result

class DifferentialEvolutionOptimizer(Optimizer):
    def minimize(self, x0, loss_fn, grad_fn=None, bounds=None):
        result = differential_evolution(loss_fn, bounds)
        return result
