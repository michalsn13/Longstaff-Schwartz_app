import numpy as np

class GBM:
    def __init__(self, S0, mu, sigma, r, values_per_year = 50):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.values_per_year = values_per_year
    def simulate_P(self, size, T, quantiles = False):
        price_moments = np.arange(1,self.values_per_year * T + 1)
        Sigma=1/self.values_per_year*np.minimum(np.tile(price_moments,(len(price_moments),1)),np.tile(price_moments.reshape(-1,1),(1,len(price_moments))))
        B = np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov = Sigma)
        sims = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * price_moments / self.values_per_year + self.sigma * B)
        return (B,sims)
    def simulate_Q(self, size, T, quantiles = False):
        price_moments = np.arange(1,self.values_per_year * T + 1)
        Sigma=1/self.values_per_year*np.minimum(np.tile(price_moments,(len(price_moments),1)),np.tile(price_moments.reshape(-1,1),(1,len(price_moments))))
        B = np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov =  Sigma)
        sims = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * price_moments / self.values_per_year + self.sigma * B)
        return (B,sims) 
    def transition_dens(self, x_curr, x_next):
        dt = 1 / self.values_per_year
        return 1/(x_next*self.sigma*np.sqrt(dt*2*np.pi))*np.exp(-(np.log(x_next/x_curr)-(self.r - 0.5*self.sigma**2)*dt)**2/(2*self.sigma**2*dt))