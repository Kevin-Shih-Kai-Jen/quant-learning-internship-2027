from pathlib import Path
import json,time
import numpy as np
from native import train,evaluate
ROOT=Path(__file__).resolve().parent

def independent(x,truth,theta,ids,tau=1.):
    g=x@theta;z=(g[None,:]-g[ids,None])/tau
    p=np.exp(-np.logaddexp(0.,-z))
    p[np.arange(len(ids)),ids]=0.
    r=1+p.sum(axis=1)
    weights=p*(1-p)/tau
    dr=weights@x-weights.sum(axis=1)[:,None]*x[ids]
    error=(r-truth[ids])/(len(x)-1)
    return float(np.mean(error**2)),np.mean(2*error[:,None]*dr/(len(x)-1),axis=0)

def independent_step(x,y,theta,ids):
    old,grad=independent(x,y,theta,ids);norm=float(grad@grad)
    if abs(grad).max()<=1e-10:return theta.copy(),0.,old,old,grad,0,1
    best=old;eta=0.;trials=0
    def test(a):
        nonlocal best,eta,trials
        loss,_=independent(x,y,theta-a*grad,ids);trials+=1
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
    return theta-eta*grad,eta,old,best,grad,trials,0 if eta else 2

def main():
    rng=np.random.default_rng(20260912);n=17
    x=np.column_stack([np.ones(n),rng.normal(size=(n,11))]);y=rng.permutation(n)+1.;theta=rng.normal(scale=.1,size=12)
    records=[]
    for ids in [np.array([3]),np.array([0,4,12]),np.arange(n)]:
        loss,g=evaluate(x,y,theta,ids);ref,rg=independent(x,y,theta,ids)
        np.testing.assert_allclose(loss,ref,atol=1e-13)
        np.testing.assert_allclose(g,rg,atol=1e-13)
        finite=np.array([(independent(x,y,theta+np.eye(12)[k]*1e-6,ids)[0]-independent(x,y,theta-np.eye(12)[k]*1e-6,ids)[0])/2e-6 for k in range(12)])
        np.testing.assert_allclose(g,finite,atol=1e-9)
    order=rng.permutation(n).astype(np.int32)
    for size in [1,5,n]:
        actual,tr,_,_,_=train(x,y,theta,order,size)
        manual=theta.copy()
        for b,start in enumerate(range(0,n,size)):
            manual,eta,old,after,grad,nt,status=independent_step(x,y,manual,order[start:start+size])
            np.testing.assert_allclose(tr[b,[2,3,4]],[old,after,eta],atol=1e-12)
            assert tr[b,7]==nt and tr[b,8]==status
        np.testing.assert_allclose(actual,manual,atol=1e-12)
        records.append({'batch_size':size,'steps':len(tr),'final_partial_batch':int(tr[-1,1])})
    # Very large scales must trigger the downward grid, without changing features.
    xx=np.zeros_like(x);xx[:,0]=1;xx[:,1]=np.arange(n)*100.
    yy=np.arange(1,n+1,dtype=float);yy[0],yy[9]=yy[9],yy[0]
    actual,tr,*_=train(xx,yy,np.zeros(12),np.arange(n,dtype=np.int32),1)
    assert ((tr[:,4]>0)&(tr[:,4]<.1)).any()
    for b in range(len(tr)):
        assert tr[b,3]<=tr[b,2]
        assert tr[b,7]>=1 or tr[b,8]==1
    # Constant features have no ranking gradient, even if target ranks differ.
    zero_x=np.zeros_like(x);zero_x[:,0]=1
    _,tr,*_=train(zero_x,y,np.zeros(12),order,1)
    assert (tr[:,8]==1).all() and (tr[:,4]==0).all()
    result={'all_checks_passed':True,'native_loss_and_gradients_match_numpy':True,'finite_difference_gradients_pass':True,
        'line_search_and_entire_sequential_update_pass_match_independent_code':records,
        'partial_batch_uses_remaining_count':True,'every_rank_uses_full_stock_universe':True,
        'downward_learning_rate_grid_exercised':True,'zero_gradient_skip_exercised':True}
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    n=2000;x=np.column_stack([np.ones(n),rng.normal(size=(n,11))]);y=rng.permutation(n)+1.;order=rng.permutation(n).astype(np.int32)
    timings=[]
    for size in [n,128,1]:
        start=time.monotonic();_,tr,*_=train(x,y,np.zeros(12),order,size);elapsed=time.monotonic()-start
        rec={'batch_size':size,'seconds_for_2000_stock_day':elapsed,'trial_evaluations':int(tr[:,7].sum())};timings.append(rec);print(json.dumps(rec),flush=True)
    (ROOT/'benchmark.json').write_text(json.dumps(timings,indent=2))

if __name__=='__main__':main()
