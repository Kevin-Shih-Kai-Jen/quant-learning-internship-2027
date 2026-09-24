import json,sys,importlib.util
import numpy as np
from features import ROOT
from native_mse import train,evaluate

def independent_step(x,y,t):
    err=x@t-y;g=2*err*x;old=err**2;norm2=g@g;best=old;rate=0.;count=0;status=1
    def trial(eta):
        nonlocal best,rate,count
        count+=1;loss=(x@(t-eta*g)-y)**2
        if np.isfinite(loss) and loss<old and loss<=old-1e-4*eta*norm2 and loss<best:best=loss;rate=eta;return True
        return False
    if max(abs(g))>1e-10:
        status=2
        if trial(.1):
            for k in range(2,11):
                if not trial(k/10):break
        else:
            ok=False
            for exp in range(-2,-7,-1):
                for digit in range(9,0,-1):
                    if trial(digit*10.**exp):ok=True;break
                if ok:break
        if rate:status=0
    return t-rate*g,g,np.r_[old,best,rate,max(abs(g)),norm2,count,status,rate==1.,g[12:],x[12:],t[12:]]

def main():
    rng=np.random.default_rng(9183);x=rng.normal(size=(29,18));x[:,0]=1;x[::2,12:]=0
    t=rng.normal(size=18)*.01;y=rng.normal(size=29)*.03;order=rng.permutation(29).astype(np.int32)
    loss,grad=evaluate(x,y,t);ref=2*x.T@(x@t-y)/len(x)
    np.testing.assert_allclose(grad,ref,atol=1e-14)
    h=1e-6;fd=np.array([(np.mean((x@(t+np.eye(18)[k]*h)-y)**2)-np.mean((x@(t-np.eye(18)[k]*h)-y)**2))/(2*h) for k in range(18)])
    np.testing.assert_allclose(grad,fd,atol=1e-10)
    new,tr,ids,ss,losses=train(x,y,t,order,1,snapshot_batches=list(range(29)))
    state=t.copy()
    for b,i in enumerate(order):
        np.testing.assert_allclose(ss[b,:18],state,atol=1e-13)
        state,g,trace=independent_step(x[i],y[i],state)
        np.testing.assert_allclose(ss[b,18:],g,atol=1e-12)
        np.testing.assert_allclose(tr[b,2:],trace,atol=1e-12)
    np.testing.assert_allclose(new,state,atol=1e-13)
    x[:,12:]=0;t[12:]=np.arange(6)+1
    new,tr,*_=train(x,y,t,order,1)
    np.testing.assert_array_equal(new[12:],t[12:]);assert (tr[:,10:16]==0).all()
    # With all added inputs zero, legacy D13 training and new D18 training
    # produce exactly the same twelve price/volume coefficients and traces.
    spec=importlib.util.spec_from_file_location('v10native',ROOT.parent/'jpx_v10_mse_sgd_financial_20260913/native_mse.py')
    v10=importlib.util.module_from_spec(spec);spec.loader.exec_module(v10)
    xx=np.column_stack([x[:,:12],np.zeros(len(x))]);tt=np.r_[t[:12],0.]
    old,otr,*_=v10.train(xx,y,tt,order,1)
    np.testing.assert_array_equal(new[:12],old[:12]);np.testing.assert_array_equal(tr[:,:10],otr[:,:10])
    ex=np.zeros((1,18));ex[:,0]=1;ex[:,-1]=10000
    new,tr,*_=train(ex,np.ones(1),np.zeros(18),np.array([0]),1);assert tr[0,8]==2 and tr[0,7]==46 and (new==0).all()
    result={'passed':True,'gradient_finite_difference_max_error':float(max(abs(grad-fd))),'sequential_batches_replayed':29,
        'all_six_zero_inputs_preserve_coefficients':True,'legacy_price_only_steps_reproduce_exactly':True,'large_raw_eps_no_acceptable_eta_skip_verified':True}
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
