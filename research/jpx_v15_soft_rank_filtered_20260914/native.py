from pathlib import Path
import ctypes as ct
import numpy as np
ROOT=Path(__file__).resolve().parent
LIB=ct.CDLL(str(ROOT/'model.dylib'))
DP=ct.POINTER(ct.c_double);IP=ct.POINTER(ct.c_int)
LIB.evaluate_rank.argtypes=[ct.c_int,DP,DP,DP,ct.c_int,ct.c_int,ct.c_double,DP,DP,ct.c_int]
LIB.evaluate_rank.restype=ct.c_double
LIB.train_day.argtypes=[ct.c_int,DP,DP,DP,IP,ct.c_int,ct.c_double,DP,DP,DP,ct.c_int]
LIB.train_day.restype=ct.c_int
TAU=1.;WORKERS=8
def dp(x):return x.ctypes.data_as(DP)
def ip(x):return x.ctypes.data_as(IP)
def initial():
    t=np.zeros(27);t[24:]=1.;return t
def coefficients(t):
    c=np.zeros(27);c[:12]=t[:12]
    for m in range(3):
        a,b,q,y,u=t[12+4*m],t[24+m],t[13+4*m],t[14+4*m],t[15+4*m]
        c[12+5*m:17+5*m]=[a,-a*b,q*b,y*b,u*b]
    return c
def project(t):
    t=t.copy();t[24:]=np.clip(t[24:],0,1);return t
def gradient(x,y,t,chosen=-1,block=-1,tau=TAU):
    # Independent NumPy sigmoid and explicit score Jacobian.
    scores=x@coefficients(t);n=len(y)
    rows=np.arange(n) if chosen<0 else np.array([chosen])
    z=(scores[None,:]-scores[rows,None])/tau
    e=np.exp(-np.abs(z));p=np.where(z>=0,1/(1+e),e/(1+e))
    p[np.arange(len(rows)),rows]=0
    ranks=1+p.sum(axis=1);err=(ranks-y[rows])/(n-1)
    w=p*(1-p)/tau
    scoregrad=(2*err[:,None]*w/(len(rows)*(n-1))).sum(axis=0)
    scoregrad[rows]-=2*err*w.sum(axis=1)/(len(rows)*(n-1))
    jac=np.zeros((n,27));jac[:,:12]=x[:,:12]
    for m in range(3):
        u,v,q,a,z=x[:,12+5*m:17+5*m].T;b=t[12+4*m:16+4*m];d=t[24+m]
        jac[:,12+4*m:16+4*m]=np.column_stack([u-d*v,d*q,d*a,d*z])
        jac[:,24+m]=-b[0]*v+b[1]*q+b[2]*a+b[3]*z
    g=jac.T@scoregrad;g[0]=0
    if block==0:g[24:]=0
    if block==1:g[:24]=0
    return float(np.mean(err**2)),g,ranks
def evaluate_native(x,y,t,chosen=-1,block=-1,tau=TAU,workers=WORKERS):
    x,y,t=[np.ascontiguousarray(a,dtype=float) for a in (x,y,t)]
    g=np.zeros(27);ranks=np.zeros(len(y))
    loss=LIB.evaluate_rank(len(y),dp(x),dp(y),dp(t),chosen,block,tau,dp(g),dp(ranks),workers)
    return loss,g,ranks
def train(x,y,t,order,full,workers=WORKERS):
    x=np.ascontiguousarray(x,dtype=float);y=np.ascontiguousarray(y,dtype=float);t=t.copy()
    order=np.ascontiguousarray(order,dtype=np.int32);n=len(y)
    assert x.shape==(n,27) and np.isfinite(x).all() and np.isfinite(y).all() and n>=3
    assert np.array_equal(np.sort(order),np.arange(n))
    trace=np.zeros((n+full,2,10));snaps=np.zeros((3+full,2,54));losses=np.zeros(3)
    status=LIB.train_day(n,dp(x),dp(y),dp(t),ip(order),full,TAU,dp(trace),dp(snaps),dp(losses),workers)
    assert status==len(trace),(status,n)
    assert np.isfinite(t).all() and np.isfinite(trace).all() and np.isfinite(losses).all()
    assert ((t[24:]>=0)&(t[24:]<=1)).all()
    return t,trace,snaps,losses
