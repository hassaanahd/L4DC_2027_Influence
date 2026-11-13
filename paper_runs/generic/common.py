"""Shared setup for all paper experiments. Design constants are computed from the NOMINAL agent pattern only."""
import numpy as np, scale as S
from certify import first_lags, certified_test, measured_only
ALPHAS=(0.05,0.1,0.2)
def design(n=3):
    """Lags and pooling weights from the nominal channel pattern and nominal gains (never from W)."""
    Phi,H,G=S.make_agents(1,np.random.default_rng(0),n,het=0.0)       # nominal agent (no heterogeneity)
    tau_d,ell_d=first_lags(Phi[0],H[0],G[0],n-1)                        # decision output
    _,ell_b=first_lags(Phi[0],H[0],G[0],n-2)                            # belief output (own-lags only)
    lag_out=1+max(tau_d[d]-l for d in tau_d for l in (ell_d[d],ell_b[d]) if l is not None)
    # nominal Lemma-1 gains g_d = sum_{a+a'=L1} c' Phi^a H Phi^a' gamma_d; optimal pooling weights are proportional to g
    c=np.eye(n)[n-1]; g={}
    for d,tau in tau_d.items():
        L1=tau-2; P=Phi[0]; g[d]=sum(c@np.linalg.matrix_power(P,a)@H[0]@np.linalg.matrix_power(P,L1-a)@G[0][:,d] for a in range(L1+1))
    w={1:1.0,0:g[0]/g[1]}
    return dict(taus={1:tau_d[1],0:tau_d[0]},lag_out=lag_out,w=w,g_nom=g)
def build(N,seed,n=3,kind='random',het=0.25,**nk):
    rng=np.random.default_rng(seed)
    W=S.net_random(N,rng,**nk) if kind=='random' else (S.net_ring(N,rng,**nk) if kind=='ring' else np.zeros((N,N)))
    Phi,H,G=S.make_agents(N,rng,n,het); A,B=S.stack(W,Phi,H,G)
    return W,A,B
def truth(W,A,B,n):
    """Evaluation only: true first lags and gains from Markov parameters."""
    g=S.gains(W,A,B,n); return {d:(g[(d,'d')][0],g[(d,'d')][1]) for d in (0,1)}
def run_ours(D,des,alphas=ALPHAS,**kw):
    info={}; Z,dec=certified_test(measured_only(D),alphas,des['taus'],des['w'],des['lag_out'],info=info,**kw)
    return Z,dec,info
