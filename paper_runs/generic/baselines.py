"""Baselines of the generic benchmark (Appendix D): output Granger and DMDc, on the same measured data as the test."""
import numpy as np
from scipy.stats import norm
def _ols_t(X,y):
    b,res,rk,sv=np.linalg.lstsq(X,y,rcond=None); r=y-X@b
    s2=r@r/(len(y)-X.shape[1]); cov=s2*np.linalg.pinv(X.T@X)
    return b/np.sqrt(np.diag(cov))
def granger_outputs(D,alpha,lags=2):
    """Output-only VAR Granger test on decisions: link j->i if any lagged y_j coefficient is significant (Bonferroni)."""
    Y=D['Yd']; T,N=Y.shape; t=np.arange(lags,T)
    X=np.concatenate([np.ones((len(t),1))]+[Y[t-l] for l in range(1,lags+1)],1)
    z=norm.ppf(1-alpha/(2*N*(N-1)*lags)); dec=np.zeros((N,N),bool)
    for i in range(N):
        tt=_ols_t(X,Y[t,i])[1:].reshape(lags,N)
        dec[i]=(np.abs(tt)>z).any(0)
    np.fill_diagonal(dec,False); return dec
def arx_dmdc(D,alpha,shared_only=False):
    """DMDc/ARX fit y(k+1)=A y(k)+B u(k)+c by least squares with shared series; link j->i if A_ij significant."""
    Y=D['Yd']; T,N=Y.shape; t=np.arange(1,T-1)
    U=D['U'][:,0,:] if shared_only else D['m'].reshape(T,-1)
    X=np.concatenate([np.ones((len(t),1)),Y[t-1],D['Yb'][t-1],U[t-1],D['ush'][t-1],D['wx'][t-1,None]],1)
    z=norm.ppf(1-alpha/(2*N*(N-1))); dec=np.zeros((N,N),bool)
    for i in range(N):
        tt=_ols_t(X,Y[t,i]); dec[i]=np.abs(tt[1:N+1])>z
    np.fill_diagonal(dec,False); return dec
