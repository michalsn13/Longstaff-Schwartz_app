import numpy as np

def prob(option, nbin,b):
  T = option.T
  sims = option.underlying.simulate_Q(b,T)
  simlen = np.shape(sims)[1] #how long a sim is, but without the start since its not simulated
  inds = np.argsort(np.argsort(sims, axis = 0), axis = 0)
  c = int(b//nbin) #some technicality without which this falls apart
  inds //= c
  inds = inds.astype(int)
  alpha = b/nbin
  betas = np.zeros((simlen, nbin, nbin))
  for tt in range(simlen - 1):
    step_array = inds[:,tt:tt+2]
    beta = np.zeros((nbin,nbin))
    np.add.at(beta, (step_array[:,0], step_array[:,1]), 1)
    betas[tt,:,:] = beta 
  probgrid = betas/alpha
  hsims = sims.copy()
  hsims.sort(axis = 0)
  return probgrid, sims, hsims


def SS(option, sims, probs, hsims):
  r = option.underlying.r
  T = option.T
  simlen = np.shape(sims)[1]
  dt = T/simlen
  nbin = np.shape(probs)[1]
  b = np.shape(sims)[0]
  Grid = np.zeros((nbin,simlen))
  alpha = b//nbin
  Time = np.arange(dt, T + dt, dt)
  hs = np.zeros((nbin, simlen)) #they will represent the payoffs of each bin 
  condition = option.barrier_ind_func(hsims, np.tile(Time,(b,1)))
  Pay = option.payoff_func(hsims, np.tile(Time,(b,1))) * condition
  Pay[condition==False] = -0.00000000001#doesnt really change the price but is used to make making plots easier
  for k in range(nbin):
    hs[k,:] = np.sum(Pay[alpha*k:alpha*(k+1),:], axis = 0)/alpha #mean payoff of each bin is calculated here
  Grid[:,-1] = hs[:,-1]
  for tt in np.arange(simlen-2,-1,-1):
    for j in range(nbin):
      prob = probs[tt, j,:]
      V = prob*Grid[:,tt+1]
      EV = sum(V)*np.exp(-r*dt)
      Grid[j,tt] = max(EV, hs[j,tt],0) 
  price = float(np.mean(Grid[:,0])*np.exp(-r*dt))
  return price, Grid, hs
