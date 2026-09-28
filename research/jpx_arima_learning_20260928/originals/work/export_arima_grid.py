"""Export aligned per-stock/day predictions for one completed grid model."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'arima_grid');p.add_argument('--p',type=int,required=True);p.add_argument('--q',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert 1<=a.p<=5 and 1<=a.q<=5
keys=np.load(a.root/'prediction_keys.npz');data=np.load(a.root/'models'/f'p{a.p}_q{a.q}'/'predictions.npz')
f=pd.DataFrame({'Date':keys['date'],'SecuritiesCode':keys['code'],'Target':keys['target'],'Score':data['score'],'Rank':data['rank'],'Fallback':data['fallback']})
f.to_csv(a.output,index=False,compression='infer');print(a.output)
