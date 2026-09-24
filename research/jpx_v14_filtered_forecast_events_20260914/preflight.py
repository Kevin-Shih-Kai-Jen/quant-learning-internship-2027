import json
import numpy as np
from native import ROOT,initial,coefficients,gradient,project,train,LIB,dp

def step(x,y,t,block):
    old,g=gradient(x,y,t);best=old;eta=0.;chosen=t.copy();count=0
    if block==0:g[24:]=0
    else:g[:24]=0
    if np.max(abs(g))<=1e-10:return t.copy(),0.,0
    def trial(rate):
        nonlocal best,eta,chosen,count
        count+=1;nt=project(t-rate*g);v=np.mean((x@coefficients(nt)-y)**2);descent=g@(t-nt)
        if np.isfinite(v) and descent>0 and v<best and v<old and v<=old-1e-4*descent:
            best=v;eta=rate;chosen=nt;return True
        return False
    if trial(.1):
        for k in range(2,11):
            if not trial(k/10):break
    else:
        stop=False
        for ex in range(-2,-7,-1):
            for k in range(9,0,-1):
                if trial(k*10.**ex):stop=True;break
            if stop:break
    return chosen,eta,count

def main():
    rng=np.random.default_rng(1729);x=np.ascontiguousarray(rng.normal(0,.3,(31,27)));x[:,0]=1
    y=np.ascontiguousarray(rng.normal(0,.02,31));t=rng.normal(0,.02,27);t[24:]=[.2,.5,.8]
    loss,g=gradient(x,y,t);ng=np.zeros(27);nl=LIB.evaluate_model(len(y),dp(x),dp(y),dp(t),dp(ng))
    np.testing.assert_allclose(g,ng,atol=1e-13);np.testing.assert_allclose(loss,nl,atol=1e-14)
    fd=[]
    for j in range(27):
        plus=t.copy();minus=t.copy();plus[j]+=1e-6;minus[j]-=1e-6
        fd.append((gradient(x,y,plus)[0]-gradient(x,y,minus)[0])/2e-6)
    np.testing.assert_allclose(g,fd,atol=1e-10,rtol=1e-7)
    order=rng.permutation(len(y)).astype(np.int32);full=5
    end,tr,snaps,losses=train(x,y,initial(),order,full);ref=initial()
    for j in range(len(tr)):
        ix=order[j:j+1] if j<len(y) else np.arange(len(y))
        for block in range(2):
            ref,eta,count=step(x[ix],y[ix],ref,block)
            np.testing.assert_allclose(eta,tr[j,block,4],atol=1e-14);assert count==tr[j,block,7]
    np.testing.assert_allclose(end,ref,atol=1e-12)
    assert losses[2]<=losses[1]+1e-14
    # Boundary projection, strongly scaled input, zero financial features.
    boundary=t.copy();boundary[24:]=[0,1,0]
    be,bt,_,_=train(x,y,boundary,order,full);br=boundary.copy()
    for j in range(len(bt)):
        ix=order[j:j+1] if j<len(y) else np.arange(len(y))
        for block in range(2):br,_,_=step(x[ix],y[ix],br,block)
    np.testing.assert_allclose(be,br,atol=1e-12)
    xx=x.copy();xx[:,12:]=0.;ee,_,_,_=train(xx,y,t,order,0)
    np.testing.assert_array_equal(ee[12:],t[12:])
    out={'passed':True,'parameter_count':27,'finite_difference_max_error':float(np.max(abs(g-fd))),
         'independent_grid_replay_substeps':4*(31+5),'discount_boundary_projection':True,'zero_financial_inputs_leave_parameters_unchanged':True}
    (ROOT/'preflight.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
