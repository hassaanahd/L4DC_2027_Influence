"""Generic benchmark, further checks (Appendix D). Usage: python run_extra.py <part>
L: lag too short / true / too long, and the lag scan (Proposition 3). LG: Lasso-Granger baseline (not in Table 5).
Same networks and noise seeds as run_paper.py."""
import numpy as np, sys, time, pickle, zlib, warnings
import scale as S
from common import *
from certify import measured_only, certified_test
from sklearn.linear_model import LassoLarsIC
warnings.filterwarnings('ignore')
CFG=S.CFG; D3=design(3); part=sys.argv[1]; res=[]; t0=time.time()
def log(*a): print(part,*a,round(time.time()-t0),flush=True)
def e1data(seed,T,cfg=CFG):
    W,A,B=build(20,seed); D=S.simulate(W,A,B,3,T,np.random.default_rng(10_000+seed*10+T//1000),cfg); return W,A,B,D
# ---------------- Lasso-Granger (BIC), same data as DMDc ----------------
def lasso_granger(R):
    Y=R['Yd']; T,N=Y.shape; t=np.arange(2,T)
    X=np.concatenate([Y[t-1],Y[t-2],R['Yb'][t-1],R['m'][t-1].reshape(len(t),-1),R['ush'][t-1],R['wx'][t-1,None]],1)
    mu=X.mean(0); sd=X.std(0)+1e-12; Xs=(X-mu)/sd; dec=np.zeros((N,N),bool)
    for i in range(N):
        b=LassoLarsIC(criterion='bic').fit(Xs,Y[t,i]).coef_
        dec[i]=(np.abs(b[:N])>0)|(np.abs(b[N:2*N])>0)
    np.fill_diagonal(dec,False); return dec
if part=='L':
    shifts={'tau0-1':{1:1,0:2},'tau0 (true)':D3['taus'],'tau0+1':{1:3,0:4}}
    Lmax=5
    for seed in range(40):
        for T in (2000,4000):
            W,A,B,D=e1data(seed,T); R=measured_only(D); r=dict(seed=seed,T=T,W=W)
            for nm,tm in shifts.items():
                Z,dec=certified_test(R,(0.1,),tm,D3['w'],D3['lag_out']); r[nm]=dec[0.1]
            # sequential search on the decision input alone: lags 1..Lmax, level alpha/Lmax each, stop at first lag with a report
            seq=None; lag_found=None
            for lag in range(1,Lmax+1):
                Z,dec=certified_test(R,(0.1/Lmax,),{1:lag},{1:1.0},D3['lag_out'])
                if dec[0.1/Lmax].any(): seq=dec[0.1/Lmax]; lag_found=lag; break
            r['seq']=seq if seq is not None else np.zeros_like(W,bool); r['seq_lag']=lag_found
            # the same single-input test at the true lag and level alpha, for reference
            Z,dec=certified_test(R,(0.1,),{1:2},{1:1.0},D3['lag_out']); r['single_true']=dec[0.1]
            res.append(r)
        log(seed)
elif part=='LG':
    for seed in range(40):
        for T in (2000,4000):
            W,A,B,D=e1data(seed,T); res.append(dict(case='main',seed=seed,T=T,W=W,lasso=lasso_granger(measured_only(D))))
        log('main',seed)
    for seed in range(15):        # ring with neighbour-shared drivers, all links moved to non-neighbours
        W,A,B=build(40,seed,3,'ring',k=4,p=1.0,wmin=0.12)
        D=S.simulate(W,A,B,3,4000,np.random.default_rng(zlib.crc32(f'S2|1.0|{seed}|4000'.encode())),dict(CFG,spatial=True,sig_r=0.6))
        res.append(dict(case='ring',seed=seed,T=4000,W=W,lasso=lasso_granger(measured_only(D)))); log('ring',seed)
    for seed in range(30):        # shared input only
        W,A,B=build(20,seed); D=S.simulate(W,A,B,3,4000,np.random.default_rng(30_000+seed),dict(CFG,shared_only=True))
        R=measured_only(D); R['m']=R['m']-D['e']; res.append(dict(case='ifac',seed=seed,T=4000,W=W,lasso=lasso_granger(R))); log('ifac',seed)
pickle.dump(res,open(f'extra_{part}.pkl','wb')); log('done')
