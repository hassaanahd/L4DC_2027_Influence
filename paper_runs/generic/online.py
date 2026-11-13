"""Certified link test with predictable (online) plug-ins: every estimated quantity used at time k
is computed only from data available before agent j's innovation at time k, so the exact guarantee holds
without sample splitting."""
import numpy as np
from math import log, sqrt, ceil
from test import PAIRS, TAU, threshold
def regressors(D,tau,i,t):
    Yd,Yb,m,ush,wx=D['Yd'],D['Yb'],D['m'],D['ush'],D['wx']
    cols=[np.ones(len(t))]
    for l in (1,2): cols+=[Yd[t-l,i],Yb[t-l,i]]
    for l in (tau,): cols+=list(Yd[t-l].T)+list(Yb[t-l].T)
    for l in range(4): cols+=list(ush[t-l].T)+[wx[t-l]]
    for l in (tau+1,): cols+=list(m[t-l].reshape(len(t),-1).T)
    for l in range(1,tau+1): cols+=[m[t-l,i,0],m[t-l,i,1]]
    return np.stack(cols,1)
def online_resid(D,tau,o,warm,every=50,ridge=1e-3,nr=True):
    """y~_{i,o}(t) = y - x(t)'beta, beta fitted on targets at times <= t - tau - 1 (refit every `every`)."""
    Y=D['Yd'] if o=='d' else D['Yb']; T,N=Y.shape; L=tau+3
    t=np.arange(L,T); R=np.full((T,N),np.nan)
    for i in range(N):
        X=regressors(D,tau,i,t) if nr else np.ones((len(t),1)); y=Y[t,i]; p=X.shape[1]
        XX=np.zeros((p,p)); Xy=np.zeros(p); last=0; beta=np.zeros(p)
        for s0 in range(0,len(t),every):
            # targets usable for times t[s0:s0+every]: those with time <= t[s0]-tau-1
            lim=np.searchsorted(t,t[s0]-tau-1,side='right')
            if lim>last:
                XX+=X[last:lim].T@X[last:lim]; Xy+=X[last:lim].T@y[last:lim]; last=lim
            if t[s0]>=warm and last>p:
                beta=np.linalg.solve(XX+ridge*np.trace(XX)/p*np.eye(p),Xy)
            sl=slice(s0,s0+every); R[t[sl],i]=y[sl]-X[sl]@beta
    return R
def online_test(D,alpha,warm=400,every=50,pairs=PAIRS,ratio=1e4,return_parts=False,fixed_w=None,nr=True,info=None,tau_map=None):
    TAUm=TAU if tau_map is None else tau_map
    T,N=D['Yd'].shape; m=D['m']
    Rs={}
    for (d,o) in pairs:
        tau=TAUm[(d,o)]
        if (tau,o) not in Rs: Rs[(tau,o)]=online_resid(D,tau,o,warm,every,nr=nr)
    K=max(TAUm[p] for p in pairs)+6
    k=np.arange(K,T-K)
    Dk=np.zeros((len(k),N,N)); Wsum=[]
    for (d,o) in pairs:
        tau=TAUm[(d,o)]; R=Rs[(tau,o)]
        M=m[k,:,d]; Yf=R[k+tau]                       # (len k, N)
        valid=~np.isnan(Yf)
        Yf=np.where(valid,Yf,0.0)
        # predictable standardised weights: w = corr / sd_m sd_y, with corr from pairs k'+tau <= k-1
        P=Yf[:,:,None]*M[:,None,:]
        a=np.zeros((len(k),N,N))
        cum=np.cumsum(P,0); cm=np.cumsum(M**2,0); cy=np.cumsum(Yf**2,0)
        for s0 in range(0,len(k),every):
            lim=s0-tau-1
            if k[s0]>=warm+K and lim>50:
                sm=np.sqrt(cm[lim]/lim); sy=np.sqrt(cy[lim]/lim)
                r=(cum[lim]/lim)/(sy[:,None]*sm[None,:])
                a[s0:s0+every]=(r if fixed_w is None else fixed_w[(d,o)])/(sy[:,None]*sm[None,:])
        Wsum.append(a)
        Dk+=a*P
    nrm=np.sqrt(sum(w**2 for w in Wsum)); nrm[nrm==0]=1
    # rescale so the pooled weight vector has unit norm (scale-free); multiply back by sd's already inside a
    Dk=Dk/nrm
    test=k>=warm+2*K
    if info is not None:
        info['n_test']=int(test.sum())
        info['sd_y']={key:np.nanstd(R[k[test]+key[0]],0) for key,R in Rs.items()}
    Dt=Dk[test]; A=Dt.sum(0); V2=(Dt**2).sum(0)
    # V^2 range fixed from the warm-up-free scale of the first `warm` test samples is not allowed;
    # use a wide fixed range around n_test * (mean D^2 over the warm-up block), computed before testing.
    q2=np.full((N,N),float(test.sum()))   # standardised terms have E[D^2] of order 1
    inr=(V2>=q2/np.sqrt(ratio))&(V2<=q2*np.sqrt(ratio))
    with np.errstate(invalid='ignore',divide='ignore'):
        Z=np.where(V2>0,A/np.sqrt(V2),0.0)
    np.fill_diagonal(Z,0)
    c,kap,Mn=threshold(N,alpha,ratio)
    dec=(np.abs(Z)>=c)&inr
    np.fill_diagonal(dec,False)
    return (Z,dec,c,inr) if not return_parts else (Z,dec,c,inr,Dt)
