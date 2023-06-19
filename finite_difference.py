import numpy as np

def FD(option,n=400,dt=np.nan,american=True):
    r = option.underlying.r
    T = option.T
    S0 = option.underlying.S0
    sigma = option.underlying.sigma

    if np.isnan(dt):
        dt = 1/(2*sigma**2 * n**2)

    k = int(T/dt + 1)
    Time = np.arange(0,T+dt,dt)
    S = np.repeat(np.arange(1,n+1).reshape((-1,1))/n * 3*S0,k,axis=1)
    ds = S[1,0] - S[0,0]
    div = option.underlying.div
    if div>0:
        div_interval = 1/option.underlying.div_freq
        if div_interval != 0:
            next_div_moment = option.underlying.next_div_moment
            while next_div_moment < T:
                S[:,Time <= next_div_moment] += div
                next_div_moment += div_interval
        div = 0

    D = np.zeros((n-2,k))
    G = np.zeros((n-2,k))
    Th = np.zeros((n-2,k))
    V = np.zeros((n,k))
    Exc = np.zeros((n,k))

    barr = option.barrier_ind_func(S,np.tile(Time,(n,1)))
    if np.all(barr == True): barr = np.ones(S.shape)
    #print(np.tile(Time,(n,1)))
    #print(S)
    #print(S.shape,np.tile(Time,(n,1)).shape)
    Pay = option.payoff_func(S,np.tile(Time,(n,1))) * barr
    V[:,k-1] = Pay[:,k-1]
    S2 = S**2

    for i in range(k-1,0,-1):
        mV = V[0:-2,i]
        pV = V[2:,i]
        oV = V[1:-1,i]
        nD = (pV - mV)/(2*ds)
        nG = (pV - 2*oV + mV)/(ds**2)
        nTh = (r*oV-(r-div)*S[1:-1,i]*nD - sigma**2/2*S2[1:-1,i]*nG)*barr[1:-1,i]
        nV = oV - dt*nTh
        D[:,i] = nD
        G[:,i] = nG
        Th[:,i] = nTh
        V[1:-1,i-1] = nV
        V[0,i-1] = barr[0,i]*(2*nV[0] - nV[1])
        V[-1,i-1] = barr[-1,i]*(2*nV[-3]-nV[-4])
        if american:
            V[:,i-1] = np.maximum(Pay[:,i-1],V[:,i-1])
            Exc[:,i-1] = (V[:,i-1] == Pay[:,i-1]) & (V[:,i-1] != 0)

    D[:,0] = (V[2:,0] - V[0:-2,0])/(2*ds)
    id_up = np.argmax(S[:,0]>=S0)
    id_dn = np.argmin(S[:,0]<=S0)-1
    val_up, val_dn = S[id_up,0], S[id_dn,0]
    alf_up = (val_up - S0)/(val_up-val_dn)
    alf_dn = (S0 - val_dn)/(val_up-val_dn)
    val = alf_up*V[id_dn,0] + alf_dn*V[id_up,0]
    Exc += 0
    Exc[barr==0] = -1

    return val,V,Exc,S
