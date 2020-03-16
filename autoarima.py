import pandas as pd
import numpy as np
import os
import glob
import pmdarima as pm
from sklearn.preprocessing import StandardScaler

traindata = pd.read_csv('.././data/train.csv')
testdata = pd.read_csv('.././data/test.csv')


# SARIMAX Model

sxmodel = pm.auto_arima(traindata['x'], exogenous=traindata[['x_sim','Vx_sim','sat_id']], start_p=3, start_q=0, test='adf', max_p=3, max_q=0, m=12, start_P=0, seasonal=True, d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
predx=sxmodel.predict(n_periods=len(testdata), exogenous=testdata[['x_sim','Vx_sim','sat_id']])

sxmodel = pm.auto_arima(traindata['y'], exogenous=traindata[['y_sim','Vy_sim','sat_id']], start_p=3, start_q=0, test='adf', max_p=3, max_q=0, m=12, start_P=0, seasonal=True, d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
predy=sxmodel.predict(clearn_periods=len(testdata), exogenous=testdata[['y_sim','Vy_sim','sat_id']])

sxmodel = pm.auto_arima(traindata['z'], exogenous=traindata[['z_sim','Vz_sim','sat_id']], start_p=3, start_q=0, test='adf', max_p=3, max_q=0, m=12, start_P=0, seasonal=True, d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
predz=sxmodel.predict(n_periods=len(testdata), exogenous=testdata[['z_sim','Vz_sim','sat_id']])

sxmodel = pm.auto_arima(traindata['Vx'], exogenous=traindata[['Vx_sim','x_sim','sat_id']], start_p=2, start_q=0, test='adf', max_p=2, max_q=0, m=12, start_P=0, seasonal=True, d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
predvx=sxmodel.predict(n_periods=len(testdata), exogenous=testdata[['Vx_sim','x_sim','sat_id']])

sxmodel = pm.auto_arima(traindata['Vy'], exogenous=traindata[['Vy_sim','y_sim','sat_id']], start_p=3, start_q=0, test='adf', max_p=3, max_q=0, m=12, start_P=0, seasonal=True, d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
predvy=sxmodel.predict(n_periods=len(testdata), exogenous=testdata[['Vy_sim','y_sim','sat_id']])

sxmodel = pm.auto_arima(traindata['Vz'], exogenous=traindata[['Vz_sim','z_sim','sat_id']], start_p=2, start_q=0, test='adf', max_p=2, max_q=0, m=12, start_P=0, seasonal=True, d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
predvz=sxmodel.predict(n_periods=len(testdata), exogenous=testdata[['Vz_sim','z_sim','sat_id']])

finalcsv1 = pd.DataFrame({"id": testdata['id'], "x": predx,'y':predy,'z':predz,'Vx':predvx,'Vy':predvy,'Vz':predvz})
finalcsv1.to_csv('.././submissions/sarimax_preds.csv', index=False)
