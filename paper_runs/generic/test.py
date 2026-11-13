"""Threshold of Theorem 2 (eq. 6; here n_p = N(N-1) is computed from N) and the regressors of Step 1 for the
generic agent of Appendix D."""
import numpy as np
from math import log, sqrt, ceil
PAIRS=[(0,'d'),(0,'b'),(1,'d')]          # (input, output) pairs with tau0 < inf
TAU={(0,'d'):3,(0,'b'):3,(1,'d'):2}
def threshold(N,alpha,ratio=400.0,kappas=np.linspace(1.02,1.6,59)):
    best=None
    for k in kappas:
        M=ceil(log(ratio)/log(k)); c=sqrt(2*k*log(2*N*(N-1)*M/alpha))
        if best is None or c<best[0]: best=(c,k,M)
    return best
def design(D,tau,t_idx):
    """Regressors (Step 1) for target time t (array of indices), lag tau.
    Own outputs at t-1,t-2; all agents' outputs at t-tau,t-tau-1; shared series and weather at t..t-3;
    all agents' measured innovations at times <= t-tau-1 (two lags)."""
    Yd,Yb,m,ush,wx=D['Yd'],D['Yb'],D['m'],D['ush'],D['wx']
    def blk(i):
        cols=[np.ones(len(t_idx))]
        for l in (1,2): cols+= [Yd[t_idx-l,i],Yb[t_idx-l,i]]
        for l in (tau,tau+1): cols+= list(Yd[t_idx-l].T)+list(Yb[t_idx-l].T)
        for l in range(4): cols+= list(ush[t_idx-l].T)+[wx[t_idx-l]]
        for l in (tau+1,tau+2): cols+= list(m[t_idx-l].reshape(len(t_idx),-1).T)
        for l in range(1,tau+1): cols+= [m[t_idx-l,i,0],m[t_idx-l,i,1]]   # receiver's own innovations (independent of sender's)
        return np.stack(cols,1)
    return blk
def cleaned(D,Ttr,tau,o):
    """Fit predictor of y_{i,o}(t) on training times, return residual series on all times (nan where undefined)."""
    Y=D['Yd'] if o=='d' else D['Yb']; T,N=Y.shape; L=tau+3
    tr=np.arange(L,Ttr); te=np.arange(Ttr+L,T)
    R=np.full((T,N),np.nan); Rtr=np.full((T,N),np.nan)
    blk_tr=design(D,tau,tr); blk_te=design(D,tau,te)
    for i in range(N):
        Xtr=blk_tr(i); b=np.linalg.lstsq(Xtr,Y[tr,i],rcond=None)[0]
        Rtr[tr,i]=Y[tr,i]-Xtr@b; R[te,i]=Y[te,i]-blk_te(i)@b
    return Rtr,R
def link_test(D,Ttr,alpha,pairs=PAIRS,ratio=400.0):
    T,N=D['Yd'].shape; m=D['m']
    Rs={}
    for (d,o) in pairs:
        tau=TAU[(d,o)]
        if (tau,o) not in Rs: Rs[(tau,o)]=cleaned(D,Ttr,tau,o)
    K=max(TAU[p] for p in pairs)+6
    ktr=np.arange(K,Ttr-K); kte=np.arange(Ttr+K,T-K)
    # pooling weights from training half: a = cov/(var m * var ytilde), per pair (i,j) and (d,o)
    Dtr=np.zeros((len(ktr),N,N)); Dte=np.zeros((len(kte),N,N))
    for (d,o) in pairs:
        tau=TAU[(d,o)]; Rtr,Rte=Rs[(tau,o)]
        Mtr=m[ktr,:,d]; Ytr=Rtr[ktr+tau]; Mte=m[kte,:,d]; Yte=Rte[kte+tau]
        cov=(Ytr.T@Mtr)/len(ktr)                       # [i,j]
        a=cov/(Mtr.var(0)[None,:]*Ytr.var(0)[:,None])
        Dtr+=a[None]*Ytr[:,:,None]*Mtr[:,None,:]
        Dte+=a[None]*Yte[:,:,None]*Mte[:,None,:]
    A=Dte.sum(0); V2=(Dte**2).sum(0); Z=A/np.sqrt(V2)
    q2=(Dtr**2).mean(0)*len(kte)
    inrange=(V2>=q2/np.sqrt(ratio))&(V2<=q2*np.sqrt(ratio))
    c,kap,M=threshold(N,alpha,ratio)
    np.fill_diagonal(Z,0)
    return Z,(np.abs(Z)>=c)&inrange,c,inrange
