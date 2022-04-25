from scipy.optimize import minimize
from scipy import integrate
import numpy as np
from typing import Callable


def make_ode_fun(
    n_regions: int,
) -> Callable:
    def fun(y: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:

        dydt = np.zeros(n_regions)
        rs = p.reshape(n_regions, n_regions)
        for ii in range(n_regions):
            for jj in range(n_regions):
                if ii != jj:
                    add = y[jj] * rs[jj, ii] - y[ii] * rs[ii, jj]
                else:
                    add = y[ii] * rs[ii, ii]
                dydt[ii] += add
        return dydt

    return fun


class ODE_solver:
    def __init__(
        self,
        time: np.ndarray,
        y: np.ndarray,
        y0: np.ndarray,
        ode_fun: Callable,
    ):

        self.time = time
        self.y = y
        self.N = self.y.shape[0]
        self.y0 = y0
        self.ode = ode_fun

    def model(self, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        y_new = integrate.odeint(self.ode, self.y0, t, args=(p,))
        return y_new

    def min_fun(self, p: np.ndarray) -> float:
        diff = self.y.flatten() - self.model(self.time, p).flatten()
        return np.mean(diff**2)

    def optim(
        self,
        p_guess: np.ndarray,
    ) -> dict:
        return minimize(self.min_fun, p_guess)
