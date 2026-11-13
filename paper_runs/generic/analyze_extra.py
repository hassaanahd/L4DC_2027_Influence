"""Scores the runs of run_extra.py and updates extra_summary.json: sections whose .pkl files are present are
recomputed, the others are kept."""
import numpy as np, pickle, json, os
import scale as S
from test import threshold
def off(N): return ~np.eye(N,dtype=bool)
def nulls(W): return (W==0)&off(len(W))
def fp(d,W): return int((d&nulls(W)).sum())
def pw(d,W): return float((d&(W>0)).sum()/(W>0).sum())
def th(W): return ((W@W)>0)&nulls(W)
m=lambda x: float(np.mean(x)); out={}
L=lambda p: pickle.load(open(p,'rb')) if os.path.exists(p) else None
R=L('extra_L.pkl')
if R:
    tab=[]
    for T in (2000,4000):
        rs=[r for r in R if r['T']==T]
        for k in ('tau0-1','tau0 (true)','tau0+1','single_true','seq'):
            tab.append(dict(T=T,test=k,FWER=m([fp(r[k],r['W'])>0 for r in rs]),false_per_net=m([fp(r[k],r['W']) for r in rs]),
                            false_twohop=m([(r[k]&th(r['W'])).sum() for r in rs]),power=m([pw(r[k],r['W']) for r in rs])))
        tab.append(dict(T=T,test='seq lag found',lags={str(l):int(sum(r['seq_lag']==l for r in rs)) for l in (None,1,2,3,4,5)}))
    out['lag']=tab
R=L('extra_LG.pkl')
if R:
    tab=[]
    for case in ('main','ring','ifac'):
        for T in sorted({r['T'] for r in R if r['case']==case}):
            rs=[r for r in R if r['case']==case and r['T']==T]
            c=dict(case=case,T=T,runs=len(rs),FWER=m([fp(r['lasso'],r['W'])>0 for r in rs]),false_per_net=m([fp(r['lasso'],r['W']) for r in rs]),
                   power=m([pw(r['lasso'],r['W']) for r in rs]),precision=m([(r['lasso']&(r['W']>0)).sum()/max(1,r['lasso'].sum()) for r in rs]),
                   false_twohop=m([(r['lasso']&th(r['W'])).sum() for r in rs]))
            if case=='ring': c['false_near']=m([(r['lasso']&(S.ring_dist(40)<=2)&nulls(r['W'])).sum() for r in rs])
            tab.append(c)
    out['lasso']=tab
prev=json.load(open('extra_summary.json')) if os.path.exists('extra_summary.json') else {}
prev.update(out); json.dump(prev,open('extra_summary.json','w'),indent=1,default=str)
for k,v in out.items():
    print('==',k)
    for r in v: print({a:(round(b,3) if isinstance(b,float) else b) for a,b in r.items()})
