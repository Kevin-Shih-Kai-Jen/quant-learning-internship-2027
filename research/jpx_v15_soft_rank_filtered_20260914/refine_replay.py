import sys,json,ctypes as ct
import numpy as np
from native import ROOT,DP,IP,dp,ip
CAP=ct.CDLL(str(ROOT/'model_checkpoints.dylib'))
CAP.train_day.argtypes=[ct.c_int,DP,DP,DP,IP,ct.c_int,ct.c_double,DP,DP,DP,ct.c_int,DP]
CAP.train_day.restype=ct.c_int
SEG=ct.CDLL(str(ROOT/'replay_segments.dylib'))
SEG.replay_segment.argtypes=[ct.c_int,DP,DP,IP,ct.c_int,ct.c_int,DP,DP,DP]
SEG.replay_segment.restype=ct.c_int

def refine(x,z):
    x=np.ascontiguousarray(x);y=np.ascontiguousarray(z['truth_rank']);order=z['order'];n=len(y);full=len(z['trace'])-n
    t=z['theta_before'].copy();tr=np.zeros_like(z['trace']);snap=np.zeros_like(z['snapshots']);loss=np.zeros(3);cp=np.zeros(((n+7)//8+full,27))
    status=CAP.train_day(n,dp(x),dp(y),dp(t),ip(order),full,1.,dp(tr),dp(snap),dp(loss),8,dp(cp))
    assert status==n+full
    for got,expected in [(t,z['theta_after']),(tr,z['trace']),(snap,z['snapshots']),(loss,z['losses'])]:np.testing.assert_array_equal(got,expected)
    starts=list(range(0,n,8))+list(range(n,n+full));ends=starts[1:]+[n+full];stats=np.zeros(5);maxstate=0.
    for i,(begin,end) in enumerate(zip(starts,ends)):
        theta=cp[i].copy();status=SEG.replay_segment(n,dp(x),dp(y),ip(order),begin,end,dp(tr),dp(theta),dp(stats))
        assert status==end-begin
        expected=cp[i+1] if i+1<len(starts) else z['theta_after']
        err=float(np.max(abs(theta-expected)/(1+abs(expected))));maxstate=max(maxstate,err)
    assert np.max(stats)<1e-7 and maxstate<1e-7
    return {'passed':True,'checkpoint_stride':8,'all_regenerated_training_arrays_bitwise_equal':True,
        'max_relative_errors_loss_before_after_gradinf_norm_descent':stats.tolist(),'max_relative_checkpoint_parameter_error':maxstate,
        'stock_steps':n,'full_steps':full}

if __name__=='__main__':
    from features import load
    f,x,groups,cal,fin=load();variant,date=sys.argv[1:3];z=np.load(ROOT/variant/'traces'/f'{date}.npz')
    result=refine(x[z['indices']],z);(ROOT/variant/f'refined_replay_{date}.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
