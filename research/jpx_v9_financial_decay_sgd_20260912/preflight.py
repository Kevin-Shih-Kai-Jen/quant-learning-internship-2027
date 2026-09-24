import sys,json
import numpy as np
from financial_features import ROOT,V8
from native_v9 import train,evaluate
sys.path.insert(0,str(V8))
from native import train as old_train
import pandas as pd
from financial_features import align_to_signals,make_events

def independent(x,truth,theta,ids,tau=1.):
    g=x@theta;z=(g[None,:]-g[ids,None])/tau
    p=np.exp(-np.logaddexp(0.,-z));p[np.arange(len(ids)),ids]=0.
    ranks=1+p.sum(axis=1);w=p*(1-p)/tau
    dr=w@x-w.sum(axis=1)[:,None]*x[ids]
    e=(ranks-truth[ids])/(len(x)-1)
    return float(np.mean(e**2)),np.mean(2*e[:,None]*dr/(len(x)-1),axis=0)

def independent_step(x,y,theta,ids):
    old,g=independent(x,y,theta,ids);norm=float(g@g)
    if abs(g).max()<=1e-10:return theta.copy(),0.,old,old,g,0,1
    best=old;eta=0.;trials=0
    def test(a):
        nonlocal best,eta,trials
        loss,_=independent(x,y,theta-a*g,ids);trials+=1
        if np.isfinite(loss) and loss<old and loss<=old-1e-4*a*norm and loss<best:
            best=loss;eta=a;return True
        return False
    if test(.1):
        for k in range(2,11):
            if not test(k/10):break
    else:
        accepted=False
        for exponent in range(-2,-7,-1):
            for digit in range(9,0,-1):
                if test(digit*10.**exponent):accepted=True;break
            if accepted:break
    return theta-eta*g,eta,old,best,g,trials,0 if eta else 2

def main():
    rng=np.random.default_rng(20260912);n=19
    x=np.column_stack([np.ones(n),rng.normal(size=(n,12))]);x[:12,-1]=0
    theta=rng.normal(scale=.1,size=13);y=rng.permutation(n)+1.;order=rng.permutation(n).astype(np.int32)
    max_fd_error=0.
    for ids in [np.array([0]),np.array([13]),np.array([1,13,18]),np.arange(n)]:
        loss,g=evaluate(x,y,theta,ids);ref,rg=independent(x,y,theta,ids)
        np.testing.assert_allclose([loss], [ref],atol=1e-13)
        np.testing.assert_allclose(g,rg,atol=1e-13)
        fd=np.array([(independent(x,y,theta+np.eye(13)[k]*1e-6,ids)[0]-independent(x,y,theta-np.eye(13)[k]*1e-6,ids)[0])/2e-6 for k in range(13)])
        max_fd_error=max(max_fd_error,float(abs(fd-g).max()));np.testing.assert_allclose(g,fd,atol=1e-9)
    for size in [1,5,n]:
        actual,tr,*_=train(x,y,theta,order,size);manual=theta.copy()
        for b,start in enumerate(range(0,n,size)):
            before=manual.copy();ids=order[start:start+size]
            manual,eta,old,after,g,trials,status=independent_step(x,y,manual,ids)
            np.testing.assert_allclose(tr[b,[2,3,4,10,11,12]],[old,after,eta,g[-1],x[ids,-1].mean(),before[-1]],atol=1e-12)
            assert tr[b,7]==trials and tr[b,8]==status
        np.testing.assert_allclose(actual,manual,atol=1e-12)
    # A zero input does not force its rank-loss coefficient gradient to zero.
    toy=np.zeros((3,13));toy[:,0]=1;toy[:,-1]=[0,1,2]
    _,gradient=evaluate(toy,np.array([1.,2.,3.]),np.zeros(13),np.array([0]))
    np.testing.assert_allclose(gradient[-1],.375,atol=1e-13)
    # An appended all-zero feature preserves every original v8 arithmetic step,
    # and its existing coefficient stays unchanged rather than becoming zero.
    xx=x.copy();xx[:,-1]=0;tt=theta.copy();tt[-1]=7.
    old,ot,*_=old_train(xx[:,:12],y,tt[:12],order,1)
    new,nt,*_=train(xx,y,tt,order,1)
    np.testing.assert_array_equal(old,new[:12]);np.testing.assert_array_equal(ot,nt[:,:10])
    assert new[-1]==7. and (nt[:,10]==0).all()
    # Exact day 1/day 10/day 11 boundaries on an explicit synthetic calendar.
    cal=pd.bdate_range('2020-01-01',periods=15)
    f=pd.DataFrame({'SecuritiesCode':1,'SignalDate':cal})
    events=pd.DataFrame([{'SecuritiesCode':1,'AvailableDate':cal[2],'AvailablePosition':2,'EventId':0,'ComparisonAvailable':True,'MarginYoYChange':.02}])
    aligned=align_to_signals(f,events,cal)
    np.testing.assert_allclose(aligned.FinancialFeature,np.r_[0.,0.,np.arange(10,0,-1)*.002,0.,0.,0.],atol=1e-15)
    result={'passed':True,'analytic_and_finite_difference_gradients':True,'max_finite_difference_error':max_fd_error,
        'single_full_partial_batch_sequential_line_search_verified':True,'zero_own_input_rank_gradient_example':float(gradient[-1]),
        'all_zero_feature_preserves_beta_and_exact_v8_updates':True,'ten_session_decay_boundaries_pass':True}
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
