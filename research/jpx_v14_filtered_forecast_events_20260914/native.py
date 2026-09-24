from pathlib import Path
import ctypes as ct
import numpy as np
ROOT=Path(__file__).resolve().parent
LIB=ct.CDLL(str(ROOT/'model.dylib'))
DP=ct.POINTER(ct.c_double);IP=ct.POINTER(ct.c_int)
LIB.evaluate_model.argtypes=[ct.c_int,DP,DP,DP,DP];LIB.evaluate_model.restype=ct.c_double
LIB.train_day.argtypes=[ct.c_int,DP,DP,DP,IP,ct.c_int,DP,DP,DP,DP];LIB.train_day.restype=ct.c_int
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
def gradient(x,y,t):
    # Independent NumPy expression, also used by replay audit.
    e=x@coefficients(t)-y;raw=2*(x.T@e)/len(x);g=np.zeros(27);g[:12]=raw[:12]
    for m in range(3):
        r=raw[12+5*m:17+5*m];b=t[12+4*m:16+4*m];d=t[24+m]
        g[12+4*m:16+4*m]=[r[0]-d*r[1],d*r[2],d*r[3],d*r[4]]
        g[24+m]=-b[0]*r[1]+b[1:]@r[2:]
    return float(np.mean(e*e)),g
def project(t):
    t=t.copy();t[24:]=np.clip(t[24:],0,1);return t
def train(x,y,t,order,full,capture=False):
    x=np.ascontiguousarray(x,dtype=float);y=np.ascontiguousarray(y,dtype=float);t=t.copy()
    order=np.ascontiguousarray(order,dtype=np.int32);n=len(y)
    assert x.shape==(n,27) and np.isfinite(x).all() and np.isfinite(y).all() and n>=3
    assert np.array_equal(np.sort(order),np.arange(n))
    trace=np.zeros((n+full,2,10));snaps=np.zeros((3+full,2,54));losses=np.zeros(3)
    checkpoints=np.zeros(((n+7)//8+full,27)) if capture else None
    status=LIB.train_day(n,dp(x),dp(y),dp(t),ip(order),full,dp(trace),dp(snaps),dp(losses),dp(checkpoints) if capture else None)
    assert status==len(trace),(status,n)
    assert np.isfinite(t).all() and np.isfinite(trace).all() and np.isfinite(losses).all()
    assert ((t[24:]>=0)&(t[24:]<=1)).all()
    return (t,trace,snaps,losses,checkpoints) if capture else (t,trace,snaps,losses)
