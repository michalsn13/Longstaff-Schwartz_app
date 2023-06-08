import numpy as np
from sklearn.linear_model import LinearRegression

def poly(X,poly_type='Laguerre'):
    n,d = X.shape
    if poly_type=='Laguerre':
        Y = np.exp(-X/2)*np.hstack((np.ones(X.shape), 1-X, 1-2*X+X**2/2)) #,1-3*X+3/2*X**2-X**3/6
        if d>1:
            for i in range(d-1):
                id = np.arange(d)[np.arange(d)!=i]
                Y = np.hstack((Y,Y[:,i:(i+1)]*Y[:,np.arange(i+1,d)], 
                               Y[:,i:(i+1)]*Y[:,id+d]))
            Y = np.hstack((Y,Y[:,-1:]*Y[:,np.arange(d,2*d-1)]))
        return Y
    else:
        return np.hstack((X,X**2,X**3))

def LS(option, b, T_idx=np.nan ,poly_type='Laguerre'):
    r = option.underlying.r
    T = option.T
    dt = 1/option.underlying.values_per_year
    Time = np.arange(dt, T + dt, dt)
    if np.any(np.isnan(T_idx)):
        T_idx = np.arange(len(Time)-1,-1,-1)
    else: 
        T_idx = np.flip(T_idx)
    dh = np.hstack((Time[T_idx[:-1]]-Time[T_idx[1:]],Time[T_idx[-1]]))
    
    #### Barrier correcion ###
    #0mb_up = b_up*np.exp(-.5826*sig*np.sqrt(T/(len(Time)-1)))
    #mb_down = b_down*np.exp(.5826*sig*np.sqrt(T/(len(Time)-1)))
    ####
    S = option.underlying.simulate_Q(b,T,False)
    condition = option.barrier_ind_func(S, np.tile(Time,(b,1)))
    Pay = option.payoff_func(S, np.tile(Time,(b,1))) * condition

    n = S.shape[0]
    k = S.shape[1]
    if len(S.shape)>2:
        d = S.shape[2]
    else:
        d = 1
        S = S.reshape((n,k,d))
    
    s0 = option.underlying.S0
    S = S/s0 ### normalization for regression
    
    v = np.exp(-r*dh)
    vP = Pay[:,-1]*v[0]
    Excercise = np.zeros((n,k),dtype=bool)
    Excercise[:,-1] = 1
    for i,t in enumerate(T_idx[1:]):
        pP = Pay[:,t]>0
        if pP.sum() != 0:
            X = poly(S[pP,t,:],poly_type=poly_type)
            Y = vP[pP]
            E_con = LinearRegression().fit(X, Y).predict(X)
            Excercise[pP,t] = E_con < Pay[pP,t]
            vP[Excercise[:,t]] = Pay[Excercise[:,t],t]
        vP *= v[i] 

    when = np.argmax(Excercise,axis=1) 
    res = Pay[np.arange(n),when]*np.exp(-r*Time[when])
    return np.mean(res), Excercise, S*s0, np.std([np.mean(res[int(n/50)*i:int(n/50)*(i+1)]) for i in range(50)])