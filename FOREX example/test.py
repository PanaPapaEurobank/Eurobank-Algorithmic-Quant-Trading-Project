import numpy as np
import pandas as pd

import sqlite3

import chunks_maker

from SDE_models import OrnsteinUhlenbeck
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model



database = sqlite3.connect("Dataset.db")
data = pd.read_sql_query("SELECT * FROM Data", database)

eurusd_series = data.iloc[:, [0, 2]]

return_list = chunks_maker.ChunksMaker.window_breaker(eurusd_series, 250)



# Ornstein - Uhlenbeck SDE Model (Linear Model Parametric Estimation)

def OU_test():
    ou_model = OrnsteinUhlenbeck()

    pred_signals = []
    real_signals = []
    for i in range(len(return_list)-1):
        ou_model.fit_process(np.asarray(return_list[i].iloc[:, 1]))
        onestep_pred = ou_model.predict(1, X_0 = return_list[i].iloc[-1, 1])
    
    
        if onestep_pred >= return_list[i].iloc[-1, 1]:
            pred_signals.append(1)
        else:
            pred_signals.append(-1)
        
        if return_list[i+1].iloc[-1, 1] >= return_list[i].iloc[-1, 1]:
            real_signals.append(1)
        else:
            real_signals.append(-1)
        
    return sum(np.asarray(real_signals) == np.asarray(pred_signals)) / len(real_signals)



# AR(1) Model Fitting

def AR_test():
    ar_model = lambda x, y: AutoReg(x, lags=y).fit()

    pred_signals2 = []
    real_signals2 = []
    for i in range(len(return_list) - 1):
        fitted_obj = ar_model(np.asarray(return_list[i].iloc[:, 1]), 1)
        onestep_pred = fitted_obj.predict(start = return_list[i].shape[0], end = return_list[i].shape[0])
    
    
        if onestep_pred >= return_list[i].iloc[-1, 1]:
            pred_signals2.append(1)
        else:
            pred_signals2.append(-1)
        
        if return_list[i+1].iloc[-1, 1] >= return_list[i].iloc[-1, 1]:
            real_signals2.append(1)
        else:
            real_signals2.append(-1)
        
        
    return sum(np.asarray(real_signals2) == np.asarray(pred_signals2)) / len(real_signals2)



# GARCH(1, 1) Model

def GARCH_test():
    garch = lambda x: arch_model(x, mean = 'Constant', vol='GARCH', p=1, o=0, q=1, rescale=False)

    pred_signals3 = []
    real_signals3 = []
    for i in range(len(return_list) - 1):
        model_obj = garch(np.asarray(return_list[i].iloc[:, 1]))
        fitted_obj = model_obj.fit()
        onestep_pred = fitted_obj.forecast(horizon = 1)
    
    
        if onestep_pred.mean['h.1'].iloc[-1] >= return_list[i].iloc[-1, 1]:
            pred_signals3.append(1)
        else:
            pred_signals3.append(-1)
        
        if return_list[i+1].iloc[-1, 1] >= return_list[i].iloc[-1, 1]:
            real_signals3.append(1)
        else:
            real_signals3.append(-1)
        
        
    return sum(np.asarray(real_signals3) == np.asarray(pred_signals3)) / len(real_signals3)



# AR(1) - GARCH(1, 1) Model

def AR_GARCH_test():
    ar_model = lambda x, y: AutoReg(x, lags=y).fit()
    garch = lambda x: arch_model(x, mean = 'Constant', vol='GARCH', p=1, o=0, q=1, rescale=False)
    
    pred_signals_ARGARCH = []
    real_signals_ARGARCH = []
    for i in range(len(return_list) - 1):
        fitted_obj = ar_model(np.asarray(return_list[i].iloc[:, 1]), 1)
        ar_residuals = fitted_obj.resid


        garch_model = garch(ar_residuals)
        garch_fitted = garch_model.fit()

        predicted_mu = fitted_obj.predict()[0]

        garch_forecast = garch_fitted.forecast(horizon=1)
        predicted_et = garch_forecast.mean['h.1'].iloc[-1]

        prediction = predicted_mu + predicted_et
    
        if prediction >= return_list[i].iloc[-1, 1]:
            pred_signals_ARGARCH.append(1)
        else:
            pred_signals_ARGARCH.append(-1)
        
        if return_list[i+1].iloc[-1, 1] >= return_list[i].iloc[-1, 1]:
            real_signals_ARGARCH.append(1)
        else:
            real_signals_ARGARCH.append(-1)
        
        
    return sum(np.asarray(real_signals_ARGARCH) == np.asarray(pred_signals_ARGARCH)) / len(real_signals_ARGARCH)









def OU_predictions():

    ou_model = OrnsteinUhlenbeck()

    pred_signals = pd.DataFrame(zip(eurusd_series.iloc[0:250, 0], [None for i in range(250)]), columns = ["Date", "Prediction"])

    real_signals = pd.DataFrame(zip(eurusd_series.iloc[0:250, 0], [None for i in range(250)]), columns = ["Date", "Actual"])

    for i in range(len(return_list)-1):
        ou_model.fit_process(np.asarray(return_list[i].iloc[:, 1]))
        onestep_pred = ou_model.predict(1, X_0 = return_list[i].iloc[-1, 1])
    
        next_date = return_list[i+1].iloc[-1, 0]
    
        if onestep_pred >= return_list[i].iloc[-1, 1]:
            pred_signal = 1
        else:
            pred_signal = -1
        
        pred_signals.loc[i+250] = [next_date, pred_signal] 
    

        if return_list[i+1].iloc[-1, 1] >= return_list[i].iloc[-1, 1]:
            real_signal = 1
        else:
            real_signal = -1
        
        real_signals.loc[i+250] = [next_date, real_signal] 
    

    ou_table = pd.merge(pred_signals, real_signals, on = "Date", how = "inner")
    
    return ou_table


def AR_GARCH_predictions():
    
    ar_model = lambda x, y: AutoReg(x, lags=y).fit()
    garch = lambda x: arch_model(x, mean = 'Constant', vol='GARCH', p=1, o=0, q=1, rescale=False)
    
    pred_signals = pd.DataFrame(zip(eurusd_series.iloc[0:250, 0], [None for i in range(250)]), columns = ["Date", "Prediction"])

    real_signals = pd.DataFrame(zip(eurusd_series.iloc[0:250, 0], [None for i in range(250)]), columns = ["Date", "Actual"])
    
    for i in range(len(return_list) - 1):
        fitted_obj = ar_model(np.asarray(return_list[i].iloc[:, 1]), 1)
        ar_residuals = fitted_obj.resid


        garch_model = garch(ar_residuals)
        garch_fitted = garch_model.fit()

        predicted_mu = fitted_obj.predict()[0]

        garch_forecast = garch_fitted.forecast(horizon=1)
        predicted_et = garch_forecast.mean['h.1'].iloc[-1]

        prediction = predicted_mu + predicted_et

        next_date = return_list[i+1].iloc[-1, 0]
        
        if prediction >= return_list[i].iloc[-1, 1]:
            pred_signal = 1
        else:
            pred_signal = -1
            
        pred_signals.loc[i+250] = [next_date, pred_signal] 
        
        if return_list[i+1].iloc[-1, 1] >= return_list[i].iloc[-1, 1]:
            real_signal = 1
        else:
            real_signal = -1
            
        real_signals.loc[i+250] = [next_date, real_signal] 
        
    ar_garch_table = pd.merge(pred_signals, real_signals, on = "Date", how = "inner")
    
    return ar_garch_table    





if __name__ == '__main__':
    
    test_pd = AR_GARCH_predictions()
    
    complete_pd = pd.merge(test_pd, eurusd_series, how = "inner", left_on = "Date", right_on = "date")
    
    complete_pd = complete_pd.T.drop_duplicates().T
    
    
    
    connection = sqlite3.connect("C:\\Users\\Babis\\Desktop\\FOREX example\\Results.db")
    
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS Data
              (Date TEXT, Prediction INT, Actual INT, Return REAL)''')
    
    connection.commit()
    
    complete_pd.to_sql('Data', connection, if_exists='replace', index = False)