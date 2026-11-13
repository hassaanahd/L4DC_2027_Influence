"""Scores the runs of run_paper.py against the planted networks (which the test never saw) and updates
paper_summary.json: sections whose .pkl files are present are recomputed, the others are kept."""
import numpy as np, pickle, json, os
from scipy.stats import norm
from test import threshold
import scale as S
from common import design
SDM=np.sqrt(1+0.33**2); DES=design(3)
def off(N): return ~np.eye(N,dtype=bool)
def nulls(W): return (W==0)&off(len(W))
def fp(dec,W): return int((dec&nulls(W)).sum())
def pw(dec,W): return float((dec&(W>0)).sum()/(W>0).sum())
def prec(dec,W): return float((dec&(W>0)).sum()/max(1,dec.sum()))
def twohop(W): return ((W@W)>0)&nulls(W)
def cthr(N,a=0.1): return threshold(N,a,1e4)[0]
def zbonf(N,a=0.1): return norm.ppf(1-a/(2*N*(N-1)))
KAPPA={'t5':9.0,'normal':3.0}
def mu_link(r,w=None,kappa=9.0,corrected=True):
    """eq. (7): the mean of Z, sqrt(n) E[D]/sqrt(E[D^2]) with D standardised as in certify.py.
    corrected: E[D^2] = s_m^2 s^2 |c|^2 + (mu4-3) sum c_d^2 a_d^2 + 2 (sum c_d a_d)^2  (sigma=1, a_d = W g_d).
    plain:     E[D^2] = s_m^2 s^2 |c|^2  (the weak-link approximation)."""
    w=DES['w'] if w is None else w; W=r['W']; wn=np.linalg.norm(list(w.values())); c={d:w[d]/wn for d in w}
    a={d:W*r['g'][d] for d in w}; s=r['sd_y'][:,None]
    num=sum(c[d]*a[d] for d in w); den2=SDM**2*s**2
    if corrected: den2=den2+(kappa-3)*sum(c[d]**2*a[d]**2 for d in w)+2*num**2
    return np.sqrt(r['n_test'])*num/np.sqrt(den2)
def pred_power(r,w=None,a=0.1,kappa=9.0,corrected=True):
    m=mu_link(r,w,kappa,corrected); c=cthr(len(r['W']),a); p=norm.cdf(m-c)+norm.cdf(-m-c); return p
def mean(x): return float(np.mean(x))
def ours_cell(rs,a=0.1):
    return dict(runs=len(rs),FWER=mean([fp(r['dec'][a],r['W'])>0 for r in rs]),false_per_net=mean([fp(r['dec'][a],r['W']) for r in rs]),
                power=mean([pw(r['dec'][a],r['W']) for r in rs]),pred=mean([pred_power(r)[r['W']>0].mean() for r in rs]) if 'g' in rs[0] else None,
                links=mean([(r['W']>0).sum() for r in rs]))
def base_cell(rs,b,a=0.1):
    return dict(FWER=mean([fp(r[b][a],r['W'])>0 for r in rs]),false_per_net=mean([fp(r[b][a],r['W']) for r in rs]),
                power=mean([pw(r[b][a],r['W']) for r in rs]),precision=mean([prec(r[b][a],r['W']) for r in rs]),
                false_twohop=mean([(r[b][a]&twohop(r['W'])).sum() for r in rs]))
L=lambda p: pickle.load(open(p,'rb')) if os.path.exists(p) else None
out={}
# ---------------- E1 ----------------
E1=L('paper_E1.pkl')
if E1:
    N=20; tab=[]
    for T in sorted({r['T'] for r in E1}):
        rs=[r for r in E1 if r['T']==T]; row=dict(T=T,n_test=int(mean([r['n_test'] for r in rs])))
        for a in (0.05,0.1,0.2):
            row[f'FWER_{a}']=mean([fp(r['dec'][a],r['W'])>0 for r in rs]); row[f'power_{a}']=mean([pw(r['dec'][a],r['W']) for r in rs])
        row['pred_0.1']=mean([pred_power(r)[r['W']>0].mean() for r in rs]); row['pred_plain_0.1']=mean([pred_power(r,corrected=False)[r['W']>0].mean() for r in rs])
        nz=lambda r,c: (np.abs(r['Z'])>=c)
        row['naive_FWER']=mean([fp(nz(r,zbonf(N)),r['W'])>0 for r in rs]); row['naive_power']=mean([pw(nz(r,zbonf(N)),r['W']) for r in rs])
        row['max_null_Z_mean']=mean([np.abs(r['Z'][nulls(r['W'])]).max() for r in rs]); row['max_null_Z_max']=float(max(np.abs(r['Z'][nulls(r['W'])]).max() for r in rs))
        row['median_link_Z']=float(np.median(np.concatenate([np.abs(r['Z'][r['W']>0]) for r in rs])))
        row['pred_median_link_Z']=float(np.median(np.concatenate([mu_link(r)[r['W']>0] for r in rs])))
        tab.append(row)
    out['E1']=dict(table=tab,c={a:cthr(N,a) for a in (0.05,0.1,0.2)},zbonf={a:zbonf(N,a) for a in (0.05,0.1,0.2)},
                   runs=len(E1),links_per_net=mean([(r['W']>0).sum() for r in E1 if r['T']==E1[0]['T']]),
                   total_false_any_alpha=int(sum(fp(r['dec'][0.2],r['W']) for r in E1)),
                   tau_true=sorted({str(r['tau_true']) for r in E1}))
    base=[]
    for T in (2000,4000):
        rs=[r for r in E1 if r['T']==T]
        if not rs: continue
        base.append(dict(T=T,method='ours',**{k:v for k,v in ours_cell(rs).items() if k!='pred'},precision=mean([prec(r['dec'][0.1],r['W']) for r in rs]),
                         false_twohop=mean([(r['dec'][0.1]&twohop(r['W'])).sum() for r in rs])))
        base.append(dict(T=T,method='naive Gaussian threshold',FWER=mean([fp(np.abs(r['Z'])>=zbonf(N),r['W'])>0 for r in rs]),
                         false_per_net=mean([fp(np.abs(r['Z'])>=zbonf(N),r['W']) for r in rs]),power=mean([pw(np.abs(r['Z'])>=zbonf(N),r['W']) for r in rs])))
        for b in ('granger','arx'): base.append(dict(T=T,method=b,**base_cell(rs,b)))
    out['baselines']=base
    # per-link calibration of eq. (7) (alpha=0.1)
    P=np.concatenate([pred_power(r)[r['W']>0] for r in E1]); O=np.concatenate([r['dec'][0.1][r['W']>0] for r in E1]).astype(float)
    cal=[]
    for lo in np.arange(0,1,0.1):
        s=(P>=lo)&(P<lo+0.1+1e-9*(lo>0.85))
        if s.sum()>=15: cal.append(dict(pred=float(P[s].mean()),obs=float(O[s].mean()),n=int(s.sum())))
    out['calibration']=cal
    P=np.concatenate([pred_power(r,corrected=False)[r['W']>0] for r in E1]); cal=[]
    for lo in np.arange(0,1,0.1):
        s_=(P>=lo)&(P<lo+0.1+1e-9*(lo>0.85))
        if s_.sum()>=15: cal.append(dict(pred=float(P[s_].mean()),obs=float(O[s_].mean()),n=int(s_.sum())))
    out['calibration_plain']=cal
    Mz=np.concatenate([mu_link(r)[r['W']>0] for r in E1]); Oz=np.concatenate([np.abs(r['Z'][r['W']>0]) for r in E1])
    out['Z_ratio_obs_over_pred_median']=float(np.median(Oz/Mz))
    out['Z_ratio_obs_over_plain_median']=float(np.median(Oz/np.concatenate([mu_link(r,corrected=False)[r['W']>0] for r in E1])))
    # placebo: every pair is a true null
    Zp=np.concatenate([r['Z_placebo'][off(N)] for r in E1])
    out['placebo']=dict(runs=len(E1),pairs=int(Zp.size),mean=float(Zp.mean()),sd=float(Zp.std()),
                        frac_abs_gt_1_96=float((np.abs(Zp)>1.96).mean()),
                        FWER_certified={a:mean([(np.abs(r['Z_placebo'][off(N)])>=cthr(N,a)).any() for r in E1]) for a in (0.05,0.1,0.2)},
                        FWER_naive={a:mean([(np.abs(r['Z_placebo'][off(N)])>=zbonf(N,a)).any() for r in E1]) for a in (0.05,0.1,0.2)},
                        FWER_unadjusted_1_96=mean([(np.abs(r['Z_placebo'][off(N)])>=1.96).any() for r in E1]),
                        max_abs=float(np.abs(Zp).max()),q999=float(np.quantile(np.abs(Zp),0.999)))
    np.save('placebo_Z.npy',Zp)
# ---------------- E1b ----------------
E1b=L('paper_E1b.pkl')
if E1b:
    st=[]
    for name in dict.fromkeys(r['stress'] for r in E1b):
        rs=[r for r in E1b if r['stress']==name]
        kap={'Gaussian innovations':3.0,'heavy tails t(3)':np.inf,'heavier tails t(2.5)':np.inf}.get(name,9.0)
        oc=ours_cell(rs); oc['pred']=mean([pred_power(r,kappa=kap)[r['W']>0].mean() for r in rs]) if np.isfinite(kap) else None
        st.append(dict(stress=name,ours=oc,granger=base_cell(rs,'granger'),arx=base_cell(rs,'arx'),
                       ours_FWER_0_2=mean([fp(r['dec'][0.2],r['W'])>0 for r in rs])))
    out['stress']=st
# ---------------- E3 ----------------
E3=L('paper_E3.pkl')
if E3:
    out['ifac']=dict(runs=len(E3),ours_reported=mean([r['dec'][0.1].sum() for r in E3]),ours_false=mean([fp(r['dec'][0.1],r['W']) for r in E3]),
                     arx=base_cell(E3,'arx'),granger=base_cell(E3,'granger'),links=mean([(r['W']>0).sum() for r in E3]))
# ---------------- E4 ----------------
E4=L('paper_E4.pkl')
if E4:
    e4=[]
    for pl in ('P1','P2','P3','P4'):
        for T in sorted({r['T'] for r in E4}):
            rs=[r for r in E4 if r['plan']==pl and r['T']==T]
            e4.append(dict(plan=pl,T=T,power=mean([pw(r['dec'][0.1],r['W']) for r in rs]),FWER=mean([fp(r['dec'][0.1],r['W'])>0 for r in rs]),
                           pred=mean([pred_power(r,r['w'])[r['W']>0].mean() for r in rs]),pred_plain=mean([pred_power(r,r['w'],corrected=False)[r['W']>0].mean() for r in rs]),
                           sd_y=mean([r['sd_y'].mean() for r in rs])))
    out['plans']=e4
# ---------------- E5 ----------------
E5=L('paper_E5.pkl')
if E5:
    out['aggregation']=[dict(Delta=D,FWER=mean([fp(r['dec'][0.1],r['W'])>0 for r in rs]),false_per_net=mean([fp(r['dec'][0.1],r['W']) for r in rs]),
                             false_twohop=mean([(r['dec'][0.1]&twohop(r['W'])).sum() for r in rs]),power=mean([pw(r['dec'][0.1],r['W']) for r in rs]),
                             max_null_Z=mean([np.abs(r['Z'][nulls(r['W'])]).max() for r in rs]),
                             max_twohop_Z=mean([np.abs(r['Z'][twohop(r['W'])]).max() for r in rs]))
                        for D in (1,2,4) for rs in [[r for r in E5 if r['Delta']==D]]]
# ---------------- blindness ----------------
B=L('paper_B.pkl')
if B:
    em=[r for r in B if r['check']=='empty']; sw=[r for r in B if r['check']=='swap']; rl=[r for r in B if r['check']=='relabel']
    out['blind']=dict(empty_runs=len(em),empty_reported_total=int(sum(r['dec'][0.1].sum() for r in em)),
                      empty_max_Z=float(max(np.abs(r['Z']).max() for r in em)),
                      swap_runs=len(sw),swap_power_own=mean([pw(r['dec'][0.1],r['W']) for r in sw]),
                      swap_false_own=mean([fp(r['dec'][0.1],r['W']) for r in sw]),
                      swap_found_of_other_only=mean([(r['dec'][0.1]&(r['Wother']>0)&(r['W']==0)).sum() for r in sw]),
                      swap_other_only_links=mean([((r['Wother']>0)&(r['W']==0)).sum() for r in sw]),
                      swap_power_if_scored_against_other=mean([pw(r['dec'][0.1],r['Wother']) for r in sw]),
                      relabel_runs=len(rl),relabel_maxdiff=float(max(r['maxdiff'] for r in rl)))
# ---------------- scaling ----------------
for part in ('S1','S2','S3','S4'):
    R=L(f'paper_{part}.pkl')
    if not R: continue
    tab=[]
    for lev in dict.fromkeys(r['level'] for r in R):
        for T in sorted({r['T'] for r in R}):
            rs=[r for r in R if r['level']==lev and r['T']==T]
            if not rs: continue
            c=dict(level=str(lev),T=T,**ours_cell(rs),N=len(rs[0]['W']),
                   twohop_pairs=mean([twohop(r['W']).sum() for r in rs]),false_twohop=mean([(r['dec'][0.1]&twohop(r['W'])).sum() for r in rs]),
                   threshold=cthr(len(rs[0]['W'])))
            if 'granger' in rs[0]: c['granger']=base_cell(rs,'granger'); c['arx']=base_cell(rs,'arx')
            if part=='S2':
                geo=lambda W:(S.ring_dist(len(W))<=2)&nulls(W); far=lambda W:(S.ring_dist(len(W))>2)&nulls(W)
                c['far_links']=mean([((r['W']>0)&(S.ring_dist(len(r['W']))>2)).sum() for r in rs])
                c['geo_nonlink_pairs']=mean([geo(r['W']).sum() for r in rs])
                for nm,f in (('geo',geo),('far',far)):
                    c['ours_false_'+nm]=mean([(r['dec'][0.1]&f(r['W'])).sum() for r in rs])
                    if 'granger' in rs[0]:
                        for b in ('granger','arx'): c[b+'_false_'+nm]=mean([(r[b][0.1]&f(r['W'])).sum() for r in rs])
            tab.append(c)
    out[part]=dict(table=tab,runs=len(R),runs_with_false_link_a01=int(sum(fp(r['dec'][0.1],r['W'])>0 for r in R)),
                   runs_with_false_link_a02=int(sum(fp(r['dec'][0.2],r['W'])>0 for r in R)),
                   tau_true=sorted({str(r['tau_true']) for r in R}))
prev=json.load(open('paper_summary.json')) if os.path.exists('paper_summary.json') else {}
prev.update(out); json.dump(prev,open('paper_summary.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=lambda x: round(float(x),3))[:20000])
