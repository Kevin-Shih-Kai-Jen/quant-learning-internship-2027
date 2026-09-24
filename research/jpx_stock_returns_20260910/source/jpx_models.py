"""Model plug-ins. Keep data calendars and trading execution out of models."""
import numpy as np
from jpx_common import FEATURES, gd_fit
LAG_FEATURES=[f'Previous{k}' for k in FEATURES]

def matrix(frame,columns):
    return np.column_stack([np.ones(len(frame)),frame[columns]])

def fit_u(current, previous, target):
    """Projected GD on scalar MSE; beta stays fixed at the raw-model train fit."""
    delta=current-previous
    a=float(np.mean(delta**2))
    b=float(np.mean(delta*(previous-target)))
    u=.5
    if a==0:
        return u, {'identifiable':False,'iterations':0,'reason':'raw predictions identical'}
    lr=.25/a
    for step in range(10000):
        gradient=2*(a*u+b)
        new_u=float(np.clip(u-lr*gradient,0.,1.))
        if abs(new_u-u)<1e-12:
            u=new_u
            break
        u=new_u
    else:
        raise RuntimeError('Projected GD for U did not converge')
    reference=float(np.clip(-b/a,0.,1.))
    assert abs(reference-u)<1e-10
    blended=u*current+(1-u)*previous
    return u, {'identifiable':True,'iterations':step+1,'learning_rate':lr,
        'gradient':float(2*(a*u+b)),'constrained_reference':reference,
        'reference_abs_difference':abs(reference-u),'training_rows':len(target),
        'train_mse_blended':float(np.mean((blended-target)**2)),
        'train_mse_raw':float(np.mean((current-target)**2)),
        'train_mse_yesterday_only':float(np.mean((previous-target)**2))}


class RawModel:
    name='raw'
    feature_names=FEATURES
    def fit(self,train,lag_train):
        self.theta,self.fit_info=gd_fit(matrix(train,FEATURES),train.Target.to_numpy())
        self.u=1.
        self.u_fit={'estimated':False,'reason':'raw model, no blending'}
        return self
    def predict(self,frame):
        raw=matrix(frame,FEATURES)@self.theta
        previous=matrix(frame,LAG_FEATURES)@self.theta
        fallback=~frame.PreviousEligible.to_numpy(dtype=bool) | ~np.isfinite(previous)
        previous=np.where(fallback,raw,previous)
        result=frame.copy()
        result['RawPrediction']=raw
        result['PreviousRawPrediction']=previous
        result['LagFallback']=fallback
        result['U']=self.u
        result['Prediction']=self.u*raw+(1-self.u)*previous
        return result

class TwoDayBlendModel(RawModel):
    name='two_day'
    def fit(self,train,lag_train):
        super().fit(train,lag_train)
        current=matrix(lag_train,FEATURES)@self.theta
        previous=matrix(lag_train,LAG_FEATURES)@self.theta
        self.u,self.u_fit=fit_u(current,previous,lag_train.Target.to_numpy())
        assert self.u_fit['train_mse_blended'] <= self.u_fit['train_mse_raw']+1e-15
        return self

class PriceVolumeSameModel:
    name='pv_same'
    feature_names=FEATURES+[f'V{k}' for k in (5,22,60)]+[f'P{k}xV{k}' for k in (5,22,60)]
    def fit(self,train,lag_train):
        self.theta,self.fit_info=gd_fit(matrix(train,self.feature_names),train.Target.to_numpy())
        self.u=1.
        self.u_fit={'estimated':False,'reason':'price-volume model, no blending'}
        return self
    def predict(self,frame):
        result=frame.copy()
        prediction=matrix(frame,self.feature_names)@self.theta
        result['Prediction']=prediction
        # Legacy runner compatibility. Neither column introduces a lag term.
        result['RawPrediction']=prediction
        result['PreviousRawPrediction']=prediction
        result['LagFallback']=False
        result['U']=1.
        return result


class PriceVolumeCrossModel(PriceVolumeSameModel):
    name='pv_cross'
    feature_names=FEATURES+[f'V{k}' for k in (5,22,60)]+[f'P{p}xV{v}' for p in (5,22,60) for v in (5,22,60)]


MODELS={'raw':RawModel,'two_day':TwoDayBlendModel,
        'pv_same':PriceVolumeSameModel,'pv_cross':PriceVolumeCrossModel}
