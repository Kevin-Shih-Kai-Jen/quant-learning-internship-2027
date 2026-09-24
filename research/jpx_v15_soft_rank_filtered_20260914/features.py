"""Load v14's frozen audited inputs without rebuilding financial events."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
V8=ROOT.parent/'jpx_v8_soft_rank_20260912'
V11=ROOT.parent/'jpx_v11_persistent_finance_std22_20260913'
V14=ROOT.parent/'jpx_v14_filtered_forecast_events_20260914'
METRICS=['NetSales','OperatingProfit','EPS']
PARTS=['ActualGrowth','ExpectedGrowth','ForecastQoQ','ForecastYoY','Revision']
FINS=[m+p for m in METRICS for p in PARTS]
VARIANTS=['sgd_only','sgd_sqrt']

def load():
    with (V8/'inputs.pkl').open('rb') as h:f,x,groups,calendar=pickle.load(h)
    std=pd.read_pickle(V11/'return_std22_features.pkl')
    x=x.copy();x[:,-2:]=std[['PR1Scaled22','VR1Scaled22']].fillna(0).to_numpy()
    fin=pd.read_pickle(V14/'financial_signal_features.pkl')
    return f,np.column_stack([x,fin[FINS].to_numpy()]),groups,calendar,fin

