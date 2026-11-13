"""Scaling study: generic number of channels n, network generators (random in-degree, ring with
long-range rewiring), spatially correlated unmeasured drivers, and a memory-light version of the
online certified test (same statistic as online.online_test, computed in receiver blocks)."""
import numpy as np
from online import online_resid
from test import threshold
BASE=dict(rho_s=0.75,rho_b=0.45,h_b=0.30,beta_bs=0.30,phi_d=0.25,h_d=0.50,kap_b=0.30,kap_s=0.40,a_s=1.0,a_d=1.0)
# ---------------- networks ----------------
def _weights(k,rng,wmin):
    while True:
        w=rng.dirichlet(np.ones(k)*2.0)
        if w.min()>=wmin: return w
def net_random(N,rng,indeg=(2,3),wmin=0.2):
    W=np.zeros((N,N))
    for i in range(N):
        k=rng.integers(indeg[0],indeg[1]+1)
        nb=rng.choice([j for j in range(N) if j!=i],k,replace=False); W[i,nb]=_weights(k,rng,wmin)
    return W
def ring_dist(N):
    d=np.abs(np.arange(N)[:,None]-np.arange(N)[None,:]); return np.minimum(d,N-d)
def net_ring(N,rng,k=4,p=0.0,wmin=0.12):
    """Agent i follows its k nearest agents on a ring; each link is rewired with prob p to a random
    agent that is NOT a geographic neighbour (ring distance > k/2)."""
    D=ring_dist(N); W=np.zeros((N,N))
    for i in range(N):
        near=[j for j in range(N) if 0<D[i,j]<=k//2]; far=[j for j in range(N) if D[i,j]>k//2]
        nb=[]
        for j in near:
            if rng.random()<p:
                cand=[f for f in far if f not in nb]; nb.append(rng.choice(cand))
            else: nb.append(j)
        W[i,nb]=_weights(len(nb),rng,wmin)
    return W
# ---------------- agents with n channels ----------------
def make_agents(N,rng,n=3,het=0.25):
    """Channel 0 = store, 1..n-2 = chain of belief channels, n-1 = decision. n=3 is the three-channel agent of Appendix D (store, belief, decision)."""
    u=lambda v: v*rng.uniform(1-het,1+het)
    Phi=[];H=[];G=[]
    for _ in range(N):
        F=np.zeros((n,n)); F[0,0]=u(BASE['rho_s'])
        for c in range(1,n-1): F[c,c]=u(BASE['rho_b']); F[c,c-1]=u(BASE['beta_bs'])
        F[n-1,0]=u(BASE['kap_s']); F[n-1,n-2]=u(BASE['kap_b']); F[n-1,n-1]=u(BASE['phi_d'])
        h=[0.0]+[u(BASE['h_b']) for _ in range(1,n-1)]+[u(BASE['h_d'])]
        g=np.zeros((n,2)); g[0,0]=u(BASE['a_s']); g[n-1,1]=u(BASE['a_d'])
        Phi.append(F); H.append(np.diag(h)); G.append(g)
    return Phi,H,G
def stack(W,Phi,H,G):
    N=len(W); n=Phi[0].shape[0]; A=np.zeros((N*n,N*n)); B=np.zeros((N*n,N*2))
    Hb=np.zeros_like(A)
    for i in range(N):
        A[i*n:(i+1)*n,i*n:(i+1)*n]=Phi[i]; B[i*n:(i+1)*n,i*2:(i+1)*2]=G[i]
    Hd=np.zeros((N*n,N*n))
    for i in range(N): Hd[i*n:(i+1)*n,i*n:(i+1)*n]=H[i]
    A+=Hd@np.kron(W,np.eye(n))
    return A,B
def gains(W,A,B,n):
    """First lag tau0 and gain g_ij (h_ij/W_ij) for (input d, output o) from Markov parameters."""
    N=len(W); ch={'d':n-1,'b':n-2}; out={}
    for d in (0,1):
        for o,c in ch.items():
            v=B[:,d::2].copy()                    # columns: input d of each agent j
            for tau in range(1,10):
                h=v[c::n,:].copy()                # rows: channel c of agent i
                np.fill_diagonal(h,0)
                if np.abs(h).max()>1e-12: break
                v=A@v
            out[(d,o)]=(tau,np.where(W>0,h/np.where(W>0,W,1),0.0),h)
    return out
# ---------------- simulation ----------------
def simulate(W,A,B,n,T,rng,cfg):
    N=len(W); burn=200; TT=T+burn; df=cfg.get('df',5)
    def sym(size):
        if cfg.get('law','t')=='t': return rng.standard_t(df,size=size)/np.sqrt(df/(df-2))
        return rng.normal(size=size)
    e=sym((TT,N,2))*cfg['sig_e']; psi=cfg.get('psi',0.5)
    if cfg.get('spatial',False):                   # unmeasured drivers shared by geographic neighbours
        Dr=ring_dist(N); Lm=(Dr<=cfg.get('spatial_w',2)).astype(float); Lm/=np.sqrt(Lm.sum(1,keepdims=True))
    else: Lm=np.eye(N)
    ua=np.zeros((TT,N,2)); ush=np.zeros((TT,2)); r=np.zeros((TT,N)); wx=np.zeros(TT)
    for t in range(1,TT):
        ua[t]=psi*ua[t-1]+e[t]
        ush[t]=0.7*ush[t-1]+cfg['sig_sh']*rng.normal(size=2)
        r[t]=0.9*r[t-1]+cfg['sig_r']*(Lm@rng.normal(size=N))
        wx[t]=0.8*wx[t-1]+cfg['sig_wx']*rng.normal()
    U=ua+ush[:,None,:]
    if cfg.get('shared_only',False): U=np.repeat(ush[:,None,:],N,1)
    x=np.zeros(N*n); Yd=np.zeros((TT,N)); Yb=np.zeros((TT,N)); unm=np.zeros(N*n)
    for t in range(TT):
        Yd[t]=x[n-1::n]+cfg['sig_vd']*rng.normal(size=N); Yb[t]=x[n-2::n]+cfg['sig_vb']*rng.normal(size=N)
        unm[0::n]=r[t]+0.8*wx[t]; unm[n-1::n]=0.5*r[t]+0.3*wx[t]
        x=A@x+B@U[t].reshape(-1)+unm+cfg['sig_w']*rng.normal(size=N*n)
    xi=cfg['sig_xi']*rng.normal(size=(TT,N,2)); s=slice(burn,TT)
    return dict(Yd=Yd[s],Yb=Yb[s],e=e[s],m=e[s]+xi[s],U=U[s],ush=ush[s],wx=wx[s])
CFG=dict(sig_e=1.0,sig_sh=0.7,sig_r=0.3,sig_wx=1.0,sig_w=0.3,sig_vd=0.1,sig_vb=0.3,sig_xi=0.33,psi=0.5,law='t',df=5)
# ---------------- memory-light online certified test ----------------
def online_test_blocks(D,alphas,tau_map,pairs=((1,'d'),(0,'d')),fixed_w={(1,'d'):1.0,(0,'d'):0.6},
                       warm=400,every=50,ratio=1e4,block=10,info=None,mode='full',ridge=1e-3,nscreen=8):
    """Identical statistic to online.online_test with fixed pooling weights, computed in blocks of receivers."""
    T,N=D['Yd'].shape; m=D['m']; Rs={}
    for (d,o) in pairs:
        tau=tau_map[(d,o)]
        if (tau,o) not in Rs: Rs[(tau,o)]=online_resid_mode(D,tau,o,warm,every,ridge,mode,nscreen,screen_T=warm)
    K=max(tau_map[p] for p in pairs)+6; k=np.arange(K,T-K); test=k>=warm+2*K
    Z=np.zeros((N,N)); V2=np.zeros((N,N))
    for i0 in range(0,N,block):
        I=np.arange(i0,min(N,i0+block)); Dk=np.zeros((len(k),len(I),N)); Ws=[]
        for (d,o) in pairs:
            tau=tau_map[(d,o)]; R=Rs[(tau,o)]
            M=m[k,:,d]; Yf=R[k+tau][:,I]; Yf=np.where(np.isnan(Yf),0.0,Yf)
            P=Yf[:,:,None]*M[:,None,:]
            cm=np.cumsum(M**2,0); cy=np.cumsum(Yf**2,0); a=np.zeros((len(k),len(I),N))
            for s0 in range(0,len(k),every):
                lim=s0-tau-1
                if k[s0]>=warm+K and lim>50:
                    sm=np.sqrt(cm[lim]/lim); sy=np.sqrt(cy[lim]/lim)
                    a[s0:s0+every]=fixed_w[(d,o)]/(sy[:,None]*sm[None,:])
            Ws.append(a); Dk+=a*P; del P
        nrm=np.sqrt(sum(w**2 for w in Ws)); nrm[nrm==0]=1; Dk/=nrm
        Dt=Dk[test]; A=Dt.sum(0); v2=(Dt**2).sum(0)
        with np.errstate(invalid='ignore',divide='ignore'): Z[I]=np.where(v2>0,A/np.sqrt(v2),0.0)
        V2[I]=v2
    np.fill_diagonal(Z,0); q2=float(test.sum())
    inr=(V2>=q2/np.sqrt(ratio))&(V2<=q2*np.sqrt(ratio))
    dec={}
    for a_ in alphas:
        c=threshold(N,a_,ratio)[0]; dd=(np.abs(Z)>=c)&inr; np.fill_diagonal(dd,False); dec[a_]=dd
    if info is not None:
        info['n_test']=int(test.sum()); info['sd_y']={key:np.nanstd(R[k[test]+key[0]],0) for key,R in Rs.items()}
    return Z,dec
# ---------------- regressors for large networks ----------------
def regressors_mode(D,tau,i,t,mode='full',nb=None):
    """full: as online.regressors (all agents' outputs/excitations). local: receiver's own lags, own excitations,
    shared series only. screen: local + outputs/excitations of the agents in nb (chosen from past data only)."""
    Yd,Yb,m,ush,wx=D['Yd'],D['Yb'],D['m'],D['ush'],D['wx']
    cols=[np.ones(len(t))]
    for l in (1,2): cols+=[Yd[t-l,i],Yb[t-l,i]]
    if mode=='full': cols+=list(Yd[t-tau].T)+list(Yb[t-tau].T)
    if mode=='screen' and nb is not None and len(nb): cols+=list(Yd[t-tau][:,nb].T)+list(Yb[t-tau][:,nb].T)
    for l in range(4): cols+=list(ush[t-l].T)+[wx[t-l]]
    if mode=='full': cols+=list(m[t-tau-1].reshape(len(t),-1).T)
    if mode=='screen' and nb is not None and len(nb): cols+=list(m[t-tau-1][:,nb].reshape(len(t),-1).T)
    for l in range(1,tau+1): cols+=[m[t-l,i,0],m[t-l,i,1]]
    return np.stack(cols,1)
def online_resid_mode(D,tau,o,warm,every=50,ridge=1e-3,mode='full',nscreen=8,screen_T=None):
    """Predictable residuals. In 'screen' mode the candidate senders for agent i are the nscreen agents whose
    outputs at lag tau best correlate with y_i over the first screen_T samples; those samples are then used only
    as warm-up (no test statistic is formed from targets in that window), so the screen is predictable."""
    Y=D['Yd'] if o=='d' else D['Yb']; T,N=Y.shape; L=tau+3
    t=np.arange(L,T); R=np.full((T,N),np.nan)
    for i in range(N):
        nb=None
        if mode=='screen':
            s=np.arange(L,screen_T); Yl=D['Yd'][s-tau]; y0=Y[s,i]
            c=np.abs(np.corrcoef(np.c_[y0,Yl].T)[0,1:]); c[i]=-1; nb=np.argsort(-c)[:nscreen]
        X=regressors_mode(D,tau,i,t,mode,nb); y=Y[t,i]; p=X.shape[1]
        XX=np.zeros((p,p)); Xy=np.zeros(p); last=0; beta=np.zeros(p)
        for s0 in range(0,len(t),every):
            lim=np.searchsorted(t,t[s0]-tau-1,side='right')
            if lim>last: XX+=X[last:lim].T@X[last:lim]; Xy+=X[last:lim].T@y[last:lim]; last=lim
            if t[s0]>=warm and last>p: beta=np.linalg.solve(XX+ridge*np.trace(XX)/p*np.eye(p),Xy)
            sl=slice(s0,s0+every); R[t[sl],i]=y[sl]-X[sl]@beta
    return R
