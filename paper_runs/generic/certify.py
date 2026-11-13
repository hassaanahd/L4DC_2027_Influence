"""Certified link test of Theorem 2, as run in the generic benchmark (Appendix D).

BLINDNESS: `certified_test` receives only the measured series (decision and belief outputs, measured
agent-specific excitations, the shared input and regional weather) plus design constants fixed before the data are
seen (lags from the agents' channel pattern, pooling weights from the nominal agent model, alpha). It never
receives the network W, the agents' matrices, the true innovations, or the unmeasured drivers.

POOLING ALIGNED ON THE OUTPUT TIME. For receiver i, sender j and output time t,
    D_t = sum_d a_d(t) m_{j,d}(t - tau_d) ytilde_i(t),
where tau_d is the first lag of input d. Under W_ij = 0, ytilde_i(t) and a_d(t) do not depend on the
sender's innovations {e_{j,d}(s), xi_{j,d}(s) : s >= t - tau_d}, so D_t is an odd function of the block of
innovations first reaching i at time t and is conditionally symmetric given everything before it.
"""
import numpy as np
from test import threshold
MEASURED=('Yd','Yb','m','ush','wx')
def measured_only(D):
    """Strip a simulation dict down to what a field study would measure."""
    return {k:np.asarray(D[k]) for k in MEASURED}
# ---------------- lags from the channel pattern only (Definition 1, Lemma 1) ----------------
def first_lags(Phi_pat,H_pat,G_pat,c_out):
    """tau0 = L1 + 2 from the common zero pattern (A2): Phi_pat[r,q] = edge q->r, H_pat[q,r] = crossing r=>q,
    G_pat[:,d] = channels input d enters, c_out = measured channel. Also returns ell = shortest lag from an
    agent's own input d to its own measured channel (used to choose which lagged outputs may be regressors)."""
    n=Phi_pat.shape[0]; P=(Phi_pat!=0).astype(int); Hm=(H_pat!=0).astype(int)
    def reach(a):          # boolean: walk of exactly a steps from q to r  -> R[r,q]
        return (np.linalg.matrix_power(P,a)>0).astype(int)
    tau,ell={},{}
    for d in range(G_pat.shape[1]):
        g=(G_pat[:,d]!=0).astype(int); best=None
        for tot in range(0,3*n):
            for a in range(tot+1):
                v=reach(a)@(Hm@(reach(tot-a)@g))
                if v[c_out]>0: best=tot; break
            if best is not None: break
        tau[d]=None if best is None else best+2
        own=[a for a in range(0,3*n) if (reach(a)@g)[c_out]>0]
        ell[d]=None if not own else own[0]+1      # u(s) enters x(s+1): one step before any walk
    return tau,ell
# ---------------- regressors (Step 1) ----------------
def _design(R,i,t,taus,lag_out,mode,nb):
    Yd,Yb,m,ush,wx=R['Yd'],R['Yb'],R['m'],R['ush'],R['wx']
    cols=[np.ones(len(t))]
    if mode=='none': return np.stack(cols,1)                          # no regressors: only a running mean
    for l in (1,2): cols+=[Yd[t-l,i],Yb[t-l,i]]                     # receiver's own recent outputs
    sel=None if mode=='full' else nb
    if mode in ('full','screen') and (sel is None or len(sel)):
        Yo=Yd[t-lag_out]; Bo=Yb[t-lag_out]
        cols+=list((Yo if sel is None else Yo[:,sel]).T)+list((Bo if sel is None else Bo[:,sel]).T)
        for d,tau in taus.items():
            Mo=m[t-tau-1,:,d]; cols+=list((Mo if sel is None else Mo[:,sel]).T)
    for l in range(4): cols+=list(ush[t-l].T)+[wx[t-l]]                # shared series, regional weather
    for d,tau in taus.items():
        for l in range(1,tau+2): cols+=[m[t-l,i,d]]                  # receiver's own excitations
    return np.stack(cols,1)
def residuals(R,taus,lag_out,out='Yd',warm=400,every=50,ridge=1e-3,mode='screen',nscreen=8):
    """ytilde_i(t) = y_i(t) - x_i(t)'beta_i(t), beta_i(t) fitted on rows t' <= t-1 only (refit every `every`).
    In 'screen' mode the extra agents for receiver i are chosen from samples before `warm` only."""
    Y=R[out]; T,N=Y.shape; L=max(max(taus.values())+2,lag_out,3)
    t=np.arange(L,T); Rt=np.full((T,N),np.nan)
    for i in range(N):
        nb=None
        if mode=='screen':
            s=np.arange(L,warm); c=np.abs(np.corrcoef(np.c_[Y[s,i],R['Yd'][s-lag_out]].T)[0,1:]); c[i]=-1
            nb=np.argsort(-c)[:nscreen]
        X=_design(R,i,t,taus,lag_out,mode,nb); y=Y[t,i]; p=X.shape[1]
        XX=np.zeros((p,p)); Xy=np.zeros(p); last=0; beta=np.zeros(p)
        for s0 in range(0,len(t),every):
            lim=s0                                   # rows with time <= t[s0]-1
            if lim>last: XX+=X[last:lim].T@X[last:lim]; Xy+=X[last:lim].T@y[last:lim]; last=lim
            if t[s0]>=warm and last>p: beta=np.linalg.solve(XX+ridge*np.trace(XX)/p*np.eye(p),Xy)
            sl=slice(s0,s0+every); Rt[t[sl],i]=y[sl]-X[sl]@beta
    return Rt
# ---------------- the test ----------------
def certified_test(R,alphas,taus,w,lag_out,out='Yd',warm=400,every=50,ratio=1e4,mode='screen',nscreen=8,
                   ridge=1e-3,block=10,pair_lags=None,info=None):
    """R: measured series only. taus: {input d: tau_d}. w: {d: pooling weight}. pair_lags: lags at which
    m_{j,d} is paired with ytilde_i(t) (default taus; a placebo uses lags below tau_d)."""
    assert set(R)==set(MEASURED), 'the test may only see measured series'
    T,N=R['Yd'].shape; m=R['m']; pl=taus if pair_lags is None else pair_lags
    Rt=residuals(R,taus,lag_out,out,warm,every,ridge,mode,nscreen)
    t0=warm+2*max(max(taus.values()),max(pl.values()))+6; t=np.arange(t0,T)      # test window fixed in advance
    q2=float(len(t)); Z=np.zeros((N,N)); V2=np.zeros((N,N)); A=np.zeros((N,N))
    for i0 in range(0,N,block):
        I=np.arange(i0,min(N,i0+block)); Y=Rt[:,I]; Y=np.where(np.isnan(Y),0.0,Y)
        cy=np.cumsum(Y**2,0); Dk=np.zeros((len(t),len(I),N)); a=[]
        for d in w:
            M=np.zeros((T,N)); M[pl[d]:]=m[:T-pl[d],:,d]                      # M[t] = m_{.,d}(t - lag)
            cm=np.cumsum(M**2,0); ad=np.zeros((len(t),len(I),N))
            for s0 in range(0,len(t),every):
                tt=t[s0]; n_=tt-1-warm                                         # predictable: data up to t-1
                sy=np.sqrt((cy[tt-1]-cy[warm])/n_); sm=np.sqrt((cm[tt-1]-cm[warm])/n_)
                ad[s0:s0+every]=w[d]/(sy[:,None]*sm[None,:])
            a.append(ad); Dk+=ad*(Y[t][:,:,None]*M[t][:,None,:])
        Dk/=np.linalg.norm(list(w.values()))          # D_t = sum_d (w_d/|w|) (m_d/s_m)(ytilde/s_y)
        A[I]=Dk.sum(0); V2[I]=(Dk**2).sum(0)
    with np.errstate(invalid='ignore',divide='ignore'): Z=np.where(V2>0,A/np.sqrt(V2),0.0)
    np.fill_diagonal(Z,0); inr=(V2>=q2/np.sqrt(ratio))&(V2<=q2*np.sqrt(ratio))
    dec={}
    for al in alphas:
        c=threshold(N,al,ratio)[0]; dd=(np.abs(Z)>=c)&inr; np.fill_diagonal(dd,False); dec[al]=dd
    if info is not None: info.update(n_test=len(t),sd_y=np.nanstd(Rt[t],0),inr=inr)
    return Z,dec
