from pathlib import Path
import json,sys
import numpy as np
from native_mse import train,evaluate
ROOT=Path(__file__).resolve().parent

def independent_step(x,y,t):
    err=x@t-y;grad=2*err*x;old=err**2;norm2=grad@grad
    best=old;rate=0.;candidates=0;status=1
    def trial(eta):
        nonlocal best,rate,candidates
        candidates+=1
        candidate=t-eta*grad;loss=(x@candidate-y)**2
        if np.isfinite(loss) and loss<old and loss<=old-1e-4*eta*norm2 and loss<best:
            best=loss;rate=eta;return True
        return False
    if np.max(abs(grad))>1e-10:
        status=2
        if trial(.1):
            for k in range(2,11):
                if not trial(k/10):break
        else:
            success=False
            for exponent in range(-2,-7,-1):
                for digit in range(9,0,-1):
                    if trial(digit*10.**exponent):success=True;break
                if success:break
        if rate:status=0
    return t-rate*grad,grad,np.array([old,best,rate,max(abs(grad)),norm2,candidates,status,rate==1.,grad[-1],x[-1],t[-1]])

def main():
    rng=np.random.default_rng(45103);x=rng.normal(size=(17,13));x[:,0]=1;x[::2,-1]=0
    y=rng.normal(size=len(x))*.03;t=rng.normal(size=13)*.01
    loss,grad=evaluate(x,y,t);np.testing.assert_allclose(loss,np.mean((x@t-y)**2),atol=1e-14)
    np.testing.assert_allclose(grad,2*x.T@(x@t-y)/len(x),atol=1e-14)
    h=1e-6;fd=np.array([(np.mean((x@(t+np.eye(13)[j]*h)-y)**2)-np.mean((x@(t-np.eye(13)[j]*h)-y)**2))/(2*h) for j in range(13)])
    np.testing.assert_allclose(grad,fd,atol=1e-10,rtol=1e-8)
    order=rng.permutation(len(x)).astype(np.int32)
    native,tr,ids,ss,losses=train(x,y,t,order,1,snapshot_batches=list(range(len(x))))
    reference=t.copy()
    for b,i in enumerate(order):
        np.testing.assert_allclose(ss[b,:13],reference,atol=1e-13)
        reference,g,trace=independent_step(x[i],y[i],reference)
        np.testing.assert_allclose(ss[b,13:],g,atol=1e-12)
        np.testing.assert_allclose(tr[b,2:],trace,atol=1e-12,rtol=1e-11)
    np.testing.assert_allclose(native,reference,atol=1e-13)
    np.testing.assert_allclose(losses,[np.mean((x@t-y)**2),np.mean((x@native-y)**2)],atol=1e-13)
    # Exact zero feature leaves beta unchanged even when beta starts nonzero.
    x[:,-1]=0;t[-1]=7
    new,tr,*_=train(x,y,t,order,1)
    assert new[-1]==7 and (tr[:,10]==0).all() and (tr[:,12]==7).all()
    assert new[0]!=t[0]  # Intercept is trainable under return MSE.
    sys.path.insert(0,str(ROOT.parent/'jpx_v7_daily_mse_20260912'))
    from run_v7 import mse_step
    _,v7grad,_,v7loss,_,_=mse_step(t[:12],x[:,:12],y,np.ones(len(x))/len(x))
    ml,mg=evaluate(x,y,t)
    np.testing.assert_allclose(mg[:12],v7grad,atol=1e-14);np.testing.assert_allclose(ml,v7loss,atol=1e-14)
    # Large raw input may have no allowed eta; retain state and record skip.
    extreme=np.ones((1,13));extreme[0,-1]=10000
    new,tr,*_=train(extreme,np.array([1.]),np.zeros(13),np.array([0]),1)
    assert tr[0,8]==2 and tr[0,7]==46 and (new==0).all()
    result={'passed':True,'finite_difference_max_error':float(max(abs(grad-fd))),'sequential_batches_checked':17,
            'v7_daily_equal_mse_gradient_matches':True,'zero_feature_beta_unchanged':True,'intercept_updates':True,'no_allowed_eta_skip_verified':True}
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
