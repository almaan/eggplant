from scipy.optimize import minimize
from scipy import integrate
import numpy as np

class ODESolver:
    def __init__(self,
                 time,
                 y,
                 y0,
                 n_regions,
                 t0 = 0,
                ):
        self.n_regions = n_regions
        self.time = time
        self.y = y
        self.N = self.y.shape[0]
        self.y0 = y0
        self.t0 = t0

    def ode(self,
            y: np.ndarray,
            t: np.ndarray,
            p: np.ndarray):

        x_new = np.zeros(self.n_regions)
        rs = p.reshape(self.n_regions,
                       self.n_regions)

        for ii in range(self.n_regions):
            for jj in range(self.n_regions):
                if ii != jj:
                    x_new[ii] += y[jj]*rs[jj,ii]
                    x_new[ii] -= y[ii]*rs[ii,jj]
                else:
                    x_new[ii] += rs[ii,ii]
        return x_new

    def model(self,t, p):
        y_new = integrate.odeint(self.ode,
                                 self.y0,
                                 t,
                                 args=(p,))
        return y_new

    def min_fun(self, p):
        y_pred = self.model(self.time, p).flatten()
        delta =  self.y.flatten() -  y_pred
        return np.mean(delta**2)

    def optim(self, p_guess):
        return minimize(self.min_fun,
                        p_guess,
                        options = dict(maxiter = 1000),
                        )
