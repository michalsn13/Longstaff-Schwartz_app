import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller


class GBM:
    def __init__(self, S0, sigma, r, div = 0, div_freq = 1, next_div_moment = 0, values_per_year = 50):
        self.S0 = S0
        self.sigma = sigma
        self.r = r
        self.div = div
        self.div_freq = div_freq
        self.next_div_moment = next_div_moment
        self.values_per_year = values_per_year
    def simulate_Q(self, size, T, quantiles = True):
        dt = 1/self.values_per_year
        time_moments = np.arange(dt, T + dt, dt)
        if quantiles:      
            B = np.array([norm.ppf(np.arange(2 * size -1, 0, -2)/(2 * size), loc = 0, scale = np.sqrt(i)) for i in time_moments]).T
            sims = self.S0 * np.array([np.exp((self.r - 0.5 * self.sigma**2) * i +  norm.ppf(np.arange(2 * size -1, 0, -2)/(2 * size), loc = 0, scale = self.sigma * np.sqrt(i))) for i in time_moments]).T
        else:
            n = int(size/2)
            W = np.random.normal(size=(n,len(time_moments)))
            W = np.cumsum(np.sqrt(dt)*W,axis=1)
            W = np.vstack((W,-W))
            sims = self.S0*np.exp((self.r - 0.5 * self.sigma**2)*time_moments.reshape((1,-1)) + self.sigma*W)
        if self.div > 0:
            div_interval = 1/self.div_freq
            if div_interval == 0:
                sims = sims * np.exp(- self.div * time_moments)
            else:
                next_div_moment = self.next_div_moment
                while next_div_moment < T:
                    sims[:,time_moments > next_div_moment] = sims[:,time_moments > next_div_moment] - self.div
                    next_div_moment += div_interval
        return sims
    def transition_dens(self, x_curr, x_next):
        dt = 1 / self.values_per_year
        denominator = x_next*self.sigma*np.sqrt(dt*2*np.pi)
        numerator = np.exp(-(np.log(x_next/x_curr)-(self.r - 0.5*self.sigma**2)*dt)**2/(2*self.sigma**2*dt))
        return (numerator/denominator).flatten()
    
class DataUnderlying:
    def __init__(self, sims_file, S0, r, div = 0, div_freq = 1, next_div_moment = 0):
        self.basis = np.array(pd.read_csv(sims_file, header = None))
        self.S0 = S0
        self.r = r
        self.div = div
        self.div_freq = div_freq
        self.check = None
        self.next_div_moment = next_div_moment
    def check_and_set_stationarity(self, sims):
        setattr(self, "check", False)
    def simulate_Q(self, size, T, quantile = False):
        old_sims = self.basis
        old_size, values_per_all = old_sims.shape
        dt = T / values_per_all
        time_moments = np.arange(dt, T + dt, dt)
        old_sims_discounted = old_sims * np.exp(- self.r * time_moments)
        self.check_and_set_stationarity(old_sims_discounted)
        if size > old_size:
            returns = (old_sims[:,1:]/old_sims[:,:-1]).flatten()
            sims = np.random.choice(returns, size = (size - old_size, values_per_all))
            sims = self.S0 * np.cumprod(sims, axis = 1)
            sims = np.concatenate((old_sims, sims))
            self.basis = sims
        else:
            sims = old_sims
        if self.div > 0:
            div_interval = 1/self.div_freq
            if div_interval == 0:
                sims = sims * np.exp(- self.div * time_moments)
            else:
                next_div_moment = self.next_div_moment
                while next_div_moment < T:
                    sims[:,time_moments > next_div_moment] = sims[:,time_moments > next_div_moment] - self.div
                    next_div_moment += div_interval
        return sims
        

        
        
    
