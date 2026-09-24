from pathlib import Path
import ctypes as ct
import numpy as np

ROOT=Path(__file__).resolve().parent
LIB=ct.CDLL(str(ROOT/'soft_rank.dylib'))
DP=ct.POINTER(ct.c_double);IP=ct.POINTER(ct.c_int)
LIB.evaluate_rank_loss.argtypes=[ct.c_int,DP,DP,DP,IP,ct.c_int,ct.c_double,DP]
LIB.evaluate_rank_loss.restype=ct.c_double
LIB.train_rank_day.argtypes=[ct.c_int,DP,DP,DP,IP,ct.c_int,ct.c_double,DP,IP,ct.c_int,DP,DP]
LIB.train_rank_day.restype=ct.c_int

def dptr(a):return a.ctypes.data_as(DP)
def iptr(a):return a.ctypes.data_as(IP)

def prepare(x,truth,theta):
    x=np.ascontiguousarray(x,dtype=np.float64);truth=np.ascontiguousarray(truth,dtype=np.float64)
    theta=np.array(theta,dtype=np.float64,copy=True,order='C')
    assert x.ndim==2 and x.shape[1]==12 and len(x)>1 and truth.shape==(len(x),) and theta.shape==(12,)
    assert np.all(x[:,0]==1) and np.isfinite(x).all() and np.isfinite(truth).all() and np.isfinite(theta).all()
    return x,truth,theta

def evaluate(x,truth,theta,ids=None,tau=1.):
    x,truth,theta=prepare(x,truth,theta)
    ids=np.arange(len(x),dtype=np.int32) if ids is None else np.ascontiguousarray(ids,dtype=np.int32)
    assert len(ids)>0 and ids.min()>=0 and ids.max()<len(x)
    gradient=np.zeros(12)
    loss=LIB.evaluate_rank_loss(len(x),dptr(x),dptr(truth),dptr(theta),iptr(ids),len(ids),tau,dptr(gradient))
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    return loss,gradient

def train(x,truth,theta,order,batch_size,tau=1.,snapshot_batches=None):
    x,truth,theta=prepare(x,truth,theta)
    order=np.ascontiguousarray(order,dtype=np.int32)
    assert np.array_equal(np.sort(order),np.arange(len(x)))
    count=(len(x)+batch_size-1)//batch_size
    snap=np.array(sorted(set([0,count//2,count-1] if snapshot_batches is None else snapshot_batches)),dtype=np.int32)
    assert len(snap)>0 and snap.min()>=0 and snap.max()<count
    trace=np.zeros((count,10));snapshots=np.zeros((len(snap),24));losses=np.zeros(2)
    status=LIB.train_rank_day(len(x),dptr(x),dptr(truth),dptr(theta),iptr(order),batch_size,tau,dptr(trace),iptr(snap),len(snap),dptr(snapshots),dptr(losses))
    assert status==count, f'Native training failed: {status}'
    assert np.isfinite(trace).all() and np.isfinite(theta).all() and np.isfinite(losses).all()
    return theta,trace,snap,snapshots,losses
