
import numpy as np

def prob(option, nbin,b=5e4):
  T = option.T
  sims = option.underlying.simulate_Q(b,T)
  length = np.shape(sims)[1]-1 #how long a sim is, but without the start since its not simulated
  inds = np.argsort(np.argsort(sims, axis = 0, kind = 'mergesort'), axis = 0, kind = 'mergesort')
  #print("inds is done") #we're about halfway done if this prints
  inds //= (b//nbin) #gives us index of a bin the number is in
  alpha = b/nbin
  betas = np.zeros((length, nbin, nbin))
  #probgrid = np.zeros((length, nbin, nbin))
  for t in range(length - 1):
    step_array = inds[:,t:t+2]
    beta = np.zeros((nbin,nbin))
    np.add.at(beta, [step_array[:,0], step_array[:,1]], 1)
    betas[t,:,:] = beta 
  probgrid = betas/alpha
  return probgrid, inds, sims

#@title Wycena klasami
# DOKONCZYC TLUMACZENIE NA KLASY: STANALEM NA OGARNIANIU CZYM JEST BARRIER IND FUNC
def price(option,sims, inds, probs):
  r = option.r
  T = option.T
  dt = T/simlen
  Time = np.arange(dt, T + dt, dt)
  nbin = np.shape(probs)[1]
  simlen = option.underlying.values_per_year
  b = np.shape(sims)[0]
  Grid = np.zeros((nbin,simlen))
  alpha = b//nbin
  hs = np.zeros((nbin, simlen)) #they will represent the payoffs of each bin 
  hsims = sims.copy()
  hsims.sort(axis = 0, kind = "mergesort")
  #PAYOFFY PONIZEJ
  condition = option.barrier_ind_func(sims, np.tile(Time,(b,1)))
  Pay = option.payoff_func(sims, np.tile(Time,(b,1))) * condition
  for k in range(nbin):
    hs[k,:] = np.sum(payoffs[alpha*k:alpha*(k+1),:], axis = 0)/alpha
  Grid[:,-1] = hs[:,-1]
  for t in np.arange(simlen-2,-1,-1):
    for j in range(nbin):
      prob = probs[t, j,:]
      V = prob*Grid[:,t+1]
      EV = sum(V)
      Grid[j,t] = max(EV, hs[j,t]) 
  return np.mean(Grid[:,0])

