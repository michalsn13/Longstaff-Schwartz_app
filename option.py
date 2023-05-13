from underlying import GBM

class Option:
    def __init__(self, underlying: GBM, payoff_func, T, barrier_ind_func = lambda X, t: True, barrier_out = True):
        self.underlying = underlying
        self.payoff_func = payoff_func
        self.T = T
        self.barrier_ind_func = barrier_ind_func
        self.barrier_out = barrier_out
    