"""Read-only reproductions on the frozen source snapshot, using synthetic CPU data."""
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch
import contextlib
import io
import json
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source_snapshot'))
import numpy as np
import pandas as pd
import torch
from torch import nn
import features
import data_processor
import deep_learning
import models
import main
import JPX_deep_learning_models as application
import utils
import evaluation

torch.set_num_threads(1)
results={}
for p in (ROOT/'source_snapshot').glob('*.py'):
    compile(p.read_text(),str(p),'exec')
results['syntax']={'compiled_modules':9,'errors':0}

# Trace only the original date helper and stop after three identical iterations.
steps=[]
class LoopStop(RuntimeError):pass

def watch(frame,event,arg):
    if frame.f_code is not utils.create_start_end_date.__code__:
        return None
    if event=='line' and frame.f_lineno==49:
        v=frame.f_locals
        steps.append([v[k].isoformat() for k in ['start_time','train_end','val_start','val_end']])
        if len(steps)==3:
            raise LoopStop('Stopped repeated, nonadvancing date splits')
    return watch

sys.settrace(watch)
try:
    utils.create_start_end_date(gap=30)
except LoopStop as e:
    results['nonadvancing_split']={'observed_three_iterations':steps,'all_identical':steps[0]==steps[1]==steps[2],'guard':str(e)}
finally:
    sys.settrace(None)

first=[pd.Timestamp(v) for v in steps[0]]
# Even one row on every calendar day cannot fit 42 observations into 30 days.
dates=pd.date_range('2017-01-01','2017-03-05',freq='D')
values=np.arange(1,len(dates)+1,dtype=float)+10
frame=pd.DataFrame({'SecuritiesCode':1001,'Date':dates.strftime('%Y-%m-%d'),
    'Open':values,'High':values+2,'Low':values-2,'Close':values+1,
    'Volume':values*100,'AdjustmentFactor':1.,'ExpectedDividend':0.,'Target':0.01})
with patch.object(application,'main_loader',return_value=frame), \
     patch.object(torch.backends.mps,'is_available',return_value=False), \
     contextlib.redirect_stdout(io.StringIO()):
    train,val,device,mean,std=application.common_process(first,application.COL_NAMES)
X_train,y_train=features.create_time_sequence(train,42,application.FEAUTRES)
X_val,y_val=features.create_time_sequence(val,42,application.FEAUTRES)
model=models.CNN_LSTM(6,64,3,100)
optimizer=torch.optim.Adam(model.parameters())
try:
    deep_learning.train(model,deep_learning.make_loader(X_train,y_train),nn.L1Loss(),optimizer,epochs=1)
except Exception as e:
    training_error=f'{type(e).__name__}: {e}'
results['thirty_day_windows']={'first_split':steps[0],'training_rows':len(train),'validation_rows':len(val),
    'training_sequences':len(X_train),'validation_sequences':len(X_val),'training_error':training_error}

# Keep the new production kernel=3 configuration; inspect the true LSTM tensors.
model=models.CNN_LSTM(6,64,3,100).eval()
shapes={}
def hook(module,args,output):
    shapes['input']=list(args[0].shape)
    shapes['output']=list(output[0].shape)
handle=model.lstm.register_forward_hook(hook)
stdout=io.StringIO()
with torch.no_grad(),contextlib.redirect_stdout(stdout):
    prediction=model(torch.ones(2,42,6))
handle.remove()
results['new_model_shape']={'actual':shapes,'prediction':list(prediction.shape),'printed':stdout.getvalue().splitlines()}

# The entrypoint still constructs kernel=42; exercise actual load_model binding.
checkpoint={'model_state_dict':model.state_dict(),'mean':[0.]*6,'std':[1.]*6,
    'feature_columns':application.FEAUTRES,'train_start':'2017-01-01','train_end':'2021-05-10'}
with patch.object(Path,'is_file',return_value=True),patch.object(torch,'load',return_value=checkpoint):
    try:
        main.use_loaded_model('synthetic_checkpoint.pt')
    except Exception as e:
        results['checkpoint_mismatch']={'exception':type(e).__name__,'message':str(e)}

# kernel=1 remains a public default and is used by existing regression tests.
with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
    try:
        models.CNN_LSTM(6,4,1,4).eval()(torch.ones(2,1,6))
    except Exception as e:
        results['pool_guard']={'input':[2,1,6],'cnn_kernel':1,'pool_kernel':2,'exception':type(e).__name__,'message':str(e)}

# Calculate reference dates separately; this does not patch or execute model_2.
start,end=datetime(2017,1,1),datetime(2021,7,1)
train_end=start+timedelta(days=30)
reference=[]
while train_end+timedelta(days=30)<=end:
    val_end=train_end+timedelta(days=30)
    reference.append([start,train_end,train_end,val_end])
    train_end=val_end
results['advancing_reference_only']={'fold_count':len(reference),'last_split':[d.isoformat() for d in reference[-1]],
    'production_checkpoint_train_end_literal':'2021-01-01','note':'Reference calculation only; production loop was not repaired.'}

bad=pd.DataFrame({'Date':['2024-01-01']*2,'Weight':[0.5,-0.5],'Target':[np.nan,0.1]})
results['missing_target_earning']={'actual':evaluation.daily_earning(bad).tolist(),'expected':'Reject missing held-position Target instead of skipping it.'}
returns=np.array([0.01,0.0100000001,0.0100000002],dtype=np.float64)
try:
    score=evaluation.calculate_sharpe(pd.DataFrame({'Date':pd.date_range('2024-01-01',periods=3),'Weight':1.,'Target':returns}))
    results['small_variance']={'actual':score}
except Exception as e:
    results['small_variance']={'exception':type(e).__name__,'message':str(e),'float64_std':float(returns.std(ddof=1))}
(ROOT/'probe_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
