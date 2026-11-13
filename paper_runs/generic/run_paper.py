"""Generic benchmark (Appendix D, Table 5). Usage: python run_paper.py <part>; SMOKE=1 for a quick small run.
The test and both baselines receive only measured series; W, the agents' matrices, the true innovations and
the unmeasured drivers are used only to simulate and, afterwards, to score."""
import numpy as np, sys, os, time, pickle, zlib
import scale as S
from common import *
from certify import measured_only, certified_test
from baselines import granger_outputs, arx_dmdc
SMOKE=bool(os.environ.get("SMOKE")); ns=lambda k: 1 if SMOKE else k
CFG=S.CFG                         # t(5) innovations
DES={n:design(n) for n in (3,4,6,8)}
def score_pack(W,A,B,n,D,T,base=False,**kw):
    Z,dec,info=run_ours(D,DES[n],**kw); tr=truth(W,A,B,n)
    r=dict(T=T,W=W,Z=Z,dec=dec,n_test=info['n_test'],sd_y=info['sd_y'],g={d:tr[d][1] for d in tr},tau_true={d:tr[d][0] for d in tr})
    if base:
        R=measured_only(D); r['granger']={a:granger_outputs(R,a) for a in ALPHAS}; r['arx']={a:arx_dmdc(R,a) for a in ALPHAS}
    return r
part=sys.argv[1]; res=[]; t0=time.time()
def log(*a): print(part,*a,round(time.time()-t0),flush=True)
if part=='E1':          # validity and power vs T, baselines, eq. (7)
    for seed in range(ns(40)):
        W,A,B=build(20,seed)
        for T in ((2000,) if SMOKE else (1000,2000,3000,4000,6000)):
            D=S.simulate(W,A,B,3,T,np.random.default_rng(10_000+seed*10+T//1000),CFG)
            r=score_pack(W,A,B,3,D,T,base=T in (2000,4000)); r['seed']=seed
            # placebo: pair the sender's excitations with the receiver at lag 1 (below tau0): every pair is a true null
            Zp,_,_=run_ours(D,DES[3],pair_lags={1:1,0:1}); r['Z_placebo']=Zp
            res.append(r)
        log(seed)
elif part=='E1b':       # stress tests, T=4000
    stress={'Gaussian innovations':dict(CFG,law='normal'),'heavy tails t(3)':dict(CFG,df=3),'heavier tails t(2.5)':dict(CFG,df=2.5),
            'shared drivers x3':dict(CFG,sig_sh=2.1,sig_wx=3.0),'unmeasured drivers x3':dict(CFG,sig_r=0.9),
            'heterogeneity 33%':'het'}
    for seed in range(ns(40)):
        for name,cfg in stress.items():
            het=0.33 if cfg=='het' else 0.25; cfg=CFG if cfg=='het' else cfg
            W,A,B=build(20,seed,het=het)
            D=S.simulate(W,A,B,3,4000,np.random.default_rng(20_000+seed),cfg)
            r=score_pack(W,A,B,3,D,4000,base=True); r.update(seed=seed,stress=name); res.append(r)
        log(seed)
elif part=='E3':        # only a shared input, as in our earlier DMDc study (not in Table 5)
    for seed in range(ns(30)):
        W,A,B=build(20,seed)
        D=S.simulate(W,A,B,3,4000,np.random.default_rng(30_000+seed),dict(CFG,shared_only=True))
        R=measured_only(D)
        # the agent-specific part of each measured input is nothing but measurement error
        R['m']=R['m']-D['e']
        Z,dec=certified_test(R,ALPHAS,DES[3]['taus'],DES[3]['w'],DES[3]['lag_out'])
        Rs=dict(R,U=np.repeat(R['ush'][:,None,:],20,1))
        res.append(dict(seed=seed,T=4000,W=W,dec=dec,arx={a:arx_dmdc(Rs,a,shared_only=True) for a in ALPHAS},
                        granger={a:granger_outputs(R,a) for a in ALPHAS})); log(seed)
elif part=='E4':        # measurement plans
    d3=DES[3]
    plans={'P1':dict(taus={0:3},w={0:1.0},out='Yb',mode='screen'),        # store excitation -> belief survey
           'P2':dict(taus={1:2},w={1:1.0},out='Yd',mode='none'),          # decision excitation -> decision, no regressors
           'P3':dict(taus={1:2},w={1:1.0},out='Yd',mode='screen'),        # decision excitation -> decision, regressors
           'P4':dict(taus=d3['taus'],w=d3['w'],out='Yd',mode='screen')}   # both excitations -> decision, regressors
    for seed in range(ns(20)):
        W,A,B=build(20,seed); tr=truth(W,A,B,3); gb=S.gains(W,A,B,3)
        for T in ((2000,) if SMOKE else (1000,2000,4000,8000)):
            D=S.simulate(W,A,B,3,T,np.random.default_rng(40_000+seed*10+T//1000),CFG); R=measured_only(D)
            for name,p in plans.items():
                info={}; Z,dec=certified_test(R,ALPHAS,p['taus'],p['w'],d3['lag_out'],out=p['out'],mode=p['mode'],info=info)
                o='d' if p['out']=='Yd' else 'b'
                res.append(dict(seed=seed,T=T,plan=name,W=W,Z=Z,dec=dec,n_test=info['n_test'],sd_y=info['sd_y'],w=p['w'],
                                g={d:gb[(d,o)][1] for d in p['taus']}))
        log(seed)
elif part=='E5':        # sampling (A4): outputs and excitations summed over Delta model periods
    for seed in range(ns(20)):
        W,A,B=build(20,seed)
        D=S.simulate(W,A,B,3,16000,np.random.default_rng(50_000+seed),CFG); R=measured_only(D)
        for Dl in (1,2,4):
            T=16000//Dl; agg=lambda X: X[:T*Dl].reshape(T,Dl,*X.shape[1:]).sum(1)
            Ra={k:agg(v) for k,v in R.items()}
            taus=DES[3]['taus'] if Dl==1 else {1:1,0:1}
            Z,dec=certified_test(Ra,ALPHAS,taus,DES[3]['w'],2)
            res.append(dict(seed=seed,Delta=Dl,W=W,Z=Z,dec=dec))
        log(seed)
elif part=='B':         # blindness checks
    # (1) empty network: no agent follows anyone
    for seed in range(ns(20)):
        W,A,B=build(20,seed,kind='empty'); D=S.simulate(W,A,B,3,4000,np.random.default_rng(60_000+seed),CFG)
        Z,dec,_=run_ours(D,DES[3]); res.append(dict(check='empty',seed=seed,W=W,Z=Z,dec=dec))
    # (2) same agents, same noise, two different networks: the recovered network follows the data
    for seed in range(ns(10)):
        rng=np.random.default_rng(seed); W1=S.net_random(20,rng); W2=S.net_random(20,rng); Phi,H,G=S.make_agents(20,rng,3)
        for k,Wk in enumerate((W1,W2)):
            A,B=S.stack(Wk,Phi,H,G); D=S.simulate(Wk,A,B,3,4000,np.random.default_rng(61_000+seed),CFG)
            Z,dec,_=run_ours(D,DES[3]); res.append(dict(check='swap',seed=seed,which=k,W=Wk,Wother=(W2,W1)[k],Z=Z,dec=dec))
    # (3) relabel the agents in the measured data: the recovered network is relabelled the same way
    for seed in range(ns(5)):
        W,A,B=build(20,seed); D=S.simulate(W,A,B,3,4000,np.random.default_rng(62_000+seed),CFG); R=measured_only(D)
        pi=np.random.default_rng(seed).permutation(20)
        Rp={k:(v[:,pi] if k in ('Yd','Yb','m') else v) for k,v in R.items()}
        d=DES[3]; Z,_=certified_test(R,(0.1,),d['taus'],d['w'],d['lag_out'],mode='full'); Zp,_=certified_test(Rp,(0.1,),d['taus'],d['w'],d['lag_out'],mode='full')
        res.append(dict(check='relabel',seed=seed,maxdiff=float(np.abs(Zp-Z[np.ix_(pi,pi)]).max())))
    log('blind')
elif part in ('S1','S2','S3','S4'):
    if part=='S1':
        cases=[('1-2',dict(indeg=(1,2),wmin=0.25)),('2-3',dict(indeg=(2,3),wmin=0.2)),('4-5',dict(indeg=(4,5),wmin=0.1)),('6-8',dict(indeg=(6,8),wmin=0.06))]
        grid=[(lev,20,3,'random',nk,seed) for seed in range(ns(15)) for lev,nk in cases]; Ts=(2000,4000,8000,16000)
    elif part=='S2':
        grid=[(p,40,3,'ring',dict(k=4,p=p,wmin=0.12),seed) for seed in range(ns(15)) for p in (0.0,0.2,0.5,1.0)]; Ts=(2000,4000,8000)
    elif part=='S3':
        grid=[(n,20,n,'random',{},seed) for seed in range(ns(15)) for n in (3,4,6,8)]; Ts=(2000,4000,8000)
    else:
        nets={20:15,50:10,100:6,200:4,500:3}
        grid=[(N,N,3,'random',{},seed) for N in ((20,100) if SMOKE else nets) for seed in range(ns(nets[N]))]; Ts=(2000,4000,8000)
    cfg=dict(CFG,spatial=True,sig_r=0.6) if part=='S2' else CFG
    for lev,N,n,kind,nk,seed in grid:
        W,A,B=build(N,seed,n,kind,**nk)
        for T in ((2000,) if SMOKE else Ts):
            D=S.simulate(W,A,B,n,T,np.random.default_rng(zlib.crc32(f'{part}|{lev}|{seed}|{T}'.encode())),cfg)
            r=score_pack(W,A,B,n,D,T,base=(T==4000 and N<=200) or SMOKE,every=50 if N<=50 else 100,block=10 if N<=200 else 4)
            r.update(level=lev,seed=seed); res.append(r)
        log(lev,seed)
pickle.dump(res,open(f'paper_{part}{"_smoke" if SMOKE else ""}.pkl','wb')); log('done')
