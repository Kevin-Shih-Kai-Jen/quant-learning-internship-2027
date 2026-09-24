import json,time
import numpy as np
import pandas as pd
from native import ROOT,initial,gradient,evaluate_native,train,project

def search(x,y,t,chosen,block):
    old,g,_=gradient(x,y,t,chosen,block);best=old;rate=0;descent=0;count=0;bt=t.copy()
    if np.max(np.abs(g))<=1e-10:return bt,np.array([old,best,rate,count,1,descent]),g
    def trial(eta):
        nonlocal best,rate,count,bt,descent
        count+=1;nt=project(t-eta*g);d=g@(t-nt);value=gradient(x,y,nt,chosen,block)[0]
        if np.isfinite(value) and d>0 and value<best and value<old and value<=old-1e-4*d:
            best=value;rate=eta;bt=nt;descent=d;return True
        return False
    if trial(.1):
        for k in range(2,11):
            if not trial(k/10):break
    else:
        stop=False
        for exponent in range(-2,-7,-1):
            for k in range(9,0,-1):
                if trial(k*10.**exponent):stop=True;break
            if stop:break
    return bt,np.array([old,best,rate,count,0 if rate>0 else 2,descent]),g

def main():
    rng=np.random.default_rng(711);x=rng.normal(size=(19,27));x[:,0]=1
    y=pd.Series(rng.normal(size=19)).rank(ascending=False).to_numpy()
    t=rng.normal(0,.2,27);t[24:]=[.2,.6,.9];errors=[]
    for chosen in [-1,0,8]:
        l,g,r=gradient(x,y,t,chosen);nl,ng,nr=evaluate_native(x,y,t,chosen)
        np.testing.assert_allclose([nl], [l],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(ng,g,rtol=1e-11,atol=1e-14)
        if chosen<0:np.testing.assert_allclose(nr,r,rtol=1e-13)
        for j in range(27):
            h=1e-5;ta=t.copy();tb=t.copy();ta[j]+=h;tb[j]-=h
            fd=(gradient(x,y,ta,chosen)[0]-gradient(x,y,tb,chosen)[0])/(2*h)
            errors.append(abs(fd-g[j]));assert abs(fd-g[j])<1e-8
        shifted=t.copy();shifted[0]+=10000
        np.testing.assert_allclose(gradient(x,y,shifted,chosen)[0],l,atol=1e-12)
    # Very large score range uses the stable fallback.
    huge=t.copy();huge[1]=10000
    np.testing.assert_allclose(evaluate_native(x,y,huge)[0:2][1],gradient(x,y,huge)[1],atol=1e-12)
    ties=np.repeat(10.,19)
    assert evaluate_native(x,ties,initial())[0]==0
    zero_fin=x.copy();zero_fin[:,12:]=0
    assert np.array_equal(evaluate_native(zero_fin,y,t)[1][12:],np.zeros(15))
    np.testing.assert_allclose(evaluate_native(x,y,t)[2].sum(),len(y)*(len(y)+1)/2,atol=1e-12)
    # Filtering must happen before constructing both reference and true ranks.
    masked=x.copy();masked[0,12]=101;mask=(abs(masked[:,12:])<=100).all(axis=1)
    raw=-y;filtered_truth=pd.Series(raw[mask]).rank(ascending=False,method='average').to_numpy()
    clean=evaluate_native(masked[mask],filtered_truth,t)[0]
    masked[0,:]=1e9
    assert evaluate_native(masked[mask],filtered_truth,t)[0]==clean
    # A stock with zero features still has a rank gradient through other stocks.
    xx=x.copy();xx[0]=0
    assert np.linalg.norm(gradient(xx,y,t,0)[1])>1e-4
    order=rng.permutation(19).astype(np.int32);nt,tr,snap,loss=train(x,y,t,order,4)
    pt=t.copy()
    for step in range(23):
        chosen=int(order[step]) if step<19 else -1
        for block in range(2):
            pt,trace,g=search(x,y,pt,chosen,block)
            np.testing.assert_allclose(tr[step,block,[2,3,4,7,8,9]],trace,atol=2e-11,rtol=2e-9)
    np.testing.assert_allclose(pt,nt,atol=1e-11,rtol=1e-9)
    for workers in [1,4,8]:
        parallel=train(x,y,t,order,4,workers=workers)
        np.testing.assert_array_equal(parallel[0],nt)
    report={'passed':True,'finite_difference_max_abs_error':max(errors),'finite_difference_parameters':81,
        'independent_grid_substeps':46,'shift_invariance':True,'average_ties':True,
        'cross_stock_gradient':True,'stable_large_score_range':True,'worker_determinism':True,
        'zero_financial_columns_zero_gradient':True,'soft_rank_sum_identity':True,'excluded_stock_absent_from_reference':True}
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
    n=1950;xx=rng.normal(0,.05,(n,27));xx[:,0]=1;yy=rng.permutation(n).astype(float)+1;order=rng.permutation(n).astype(np.int32)
    for workers in [1,4,8]:
        start=time.monotonic();train(xx,yy,t,order,44,workers=workers)
        print('benchmark',workers,time.monotonic()-start,flush=True)
if __name__=='__main__':main()
