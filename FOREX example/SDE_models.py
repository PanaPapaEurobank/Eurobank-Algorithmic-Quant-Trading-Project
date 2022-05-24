import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


class BrownianMotion:
    
    def __init__(self, random_seed = None):
        self.random_seed = random_seed
    
    def get_dW(self, T):
        np.random.seed = self.random_seed
        return np.random.normal(0.0, 1.0, T)
    
    def get_W(self, T):
        dW = self.get_dW(T)
        dW_cumsum = dW.cumsum()
        return np.insert(dW_cumsum, 0, 0)[:-1]
    
    
class OrnsteinUhlenbeck:
    
    def __init__(self, X_0 = None):
        self.X_0 = X_0
        self.params = [0, 0, 0]
    
    
    def update_X_0(self):
        if self.X_0 == None:
            self.X_0 = self.params[2]


    def appr_integral(self, t, dW):
        exp_alpha_s = np.exp(self.params[0] * t)
        integral_W = np.cumsum(exp_alpha_s * dW)
        return np.insert(integral_W, 0, 0)[:-1]


    def fit_process(self, X_t):
        y = np.diff(X_t)
        X = X_t[:-1].reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        self.params[0] = -reg.coef_[0]
        self.params[2] = reg.intercept_ / self.params[0]
        y_hat = reg.predict(X)
        self.params[1] = np.std(y - y_hat)


    def predict(self, T, random_seed = None, X_0 = None):
        if X_0 != None:
            self.X_0 = X_0
        
        t = np.arange(T, dtype=np.longdouble) 
        exp_alpha_t = np.exp(-self.params[0] * t)
        dW = BrownianMotion(random_seed).get_dW(T)
        integral_W = self.appr_integral(t, dW)
        self.update_X_0()
        return (
            self.X_0 * exp_alpha_t
            + self.params[2] * (1 - exp_alpha_t)
            + self.params[0] * exp_alpha_t * integral_W
            )