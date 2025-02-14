import os
import tempfile
import joblib
os.environ["OMP_NUM_THREADS"] = "16"  
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "16"
os.environ['PYTHONHASHSEED']=str(1)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, PReLU, ConvLSTM1D, Dropout, Input
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import pandas as pd
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import torch
import random
from functools import reduce
from sklearn.model_selection import GridSearchCV


#tf.config.threading.set_intra_op_parallelism_threads(16)  # Defina o número de threads para operações internas
#tf.config.threading.set_inter_op_parallelism_threads(16)  

tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)


# Função para baixar os dados históricos
def retrieve_stock_data(ticker, start_dt):
    data = yf.download(ticker, start=start_dt)
    data.drop(columns=['Volume','High', 'Low', 'Open'], inplace=True)
    return data  # Apenas o preço de fechamento

# Exemplo para obter os dados históricos de PETR4
ticker = 'PETR4.SA'
data = retrieve_stock_data(ticker, '2018-01-01')

# Função para criar os conjuntos de dados de treino e teste
def format_timeseries_dataset(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])  # Últimos N registros
        y.append(data[i, 0])  # Próximo valor
    X, y = np.array(X), np.array(y)
    return X, y

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)
temp_dir = tempfile.TemporaryDirectory()
temp_artifact_dir = os.path.join(temp_dir.name,'scaler')
print(temp_artifact_dir)
os.mkdir(temp_artifact_dir)

joblib.dump(scaler, os.path.join(temp_artifact_dir,'scaler.joblib'))


train_data, test_data = train_test_split(scaled_data, test_size=0.3, random_state=1, shuffle=False)

lookback = 14
X_train, y_train = format_timeseries_dataset(train_data, lookback)
X_test, y_test = format_timeseries_dataset(test_data, lookback)


def create_model(units: int = 50, learning_rate: float= 0.0001):
    model = Sequential()
    model.add(Input((1, 1)))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate) , loss='mean_squared_error', metrics=['mae'])

    return model

model = create_model(units=100, learning_rate=0.0007)
model_history = model.fit(X_train, y_train, epochs=140, batch_size=32, validation_data=(X_test, y_test))


data_splits = 5
tscv = TimeSeriesSplit(n_splits=data_splits)
mses = []
losses = []

params = {"batch_size": [32, 38, 42, 47, 52, 60, 75], 
          "epochs": [100, 140, 200, 300], 
          "loopback": [10, 14, 20, 30], 
          "learning_rate": [0.0007, 0.0005, 0.0003, 0.0009],
          "units": [10, 30, 40, 50, 100]}


def grid_search_build(params):
    keys = params.keys()
    key_pointer = [None] * len(keys)
    max = list(map(lambda k: len(params[k]), keys))
    matrix = []
    
    total_combinations = reduce(lambda a,b : a*b, max)
    
    for index,k in enumerate(keys):
        if index == 0:
            key_pointer[index] = max[index]
        elif index == 1:
            key_pointer[index] = max[0]
        else:
            key_pointer[index] = max[index-1] * key_pointer[index-1]
                
    
    for i in range(0,total_combinations):
        combination = {}
        for index, key in enumerate(keys):
            if index == 0:
                combination[key] = params[key][(i % max[index])]
            elif index == 1:
                combination[key] = params[key][(i // (key_pointer[index])) % max[index]]
            else:
                combination[key] = params[key][(i // key_pointer[index]) % max[index]]
                
        matrix.append(combination)
    return matrix
            

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment('LSTM Stock Predictions')

for bs in params["batch_size"]:
    for ep in params["epochs"]:
        for lb in params["loopback"]:
            for lr in params["learning_rate"]:
                for un in params["units"]:
                    parameters = {"batch_size": bs, "epochs": ep, "loopback": lb, "units": un, "learning_rate": lr}
                    print(parameters)

                    with mlflow.start_run():
                        mlflow.log_params(parameters)
                        mlflow.keras.autolog()
                        #model = create_model(learning_rate=lr, units=un)
                        cross_validation_mses = []
                        cross_validation_rmses = []
                        cross_validation_mapes = []
                        signatures = []

                        for train_idx, test_idx in tscv.split(scaled_data):

                            train_data, test_data = scaled_data[train_idx], scaled_data[test_idx]
                            
                            X_train, y_train = format_timeseries_dataset(train_data, lb)
                            X_test, y_test = format_timeseries_dataset(test_data, lb)
                            
                            model = create_model(learning_rate=lr, units=un)
                            model.fit(X_train, y_train, epochs=ep, batch_size=bs, validation_data=(X_test, y_test), shuffle=False)

                            y_pred = model.predict(X_test)
                            #loss = model.evaluate(X_test, y_test)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = root_mean_squared_error(y_test, y_pred)
                            mape = mean_absolute_percentage_error(y_test, y_pred)
                            
                            cross_validation_mses.append(mse)
                            cross_validation_rmses.append(rmse)
                            cross_validation_mapes.append(mape)
                            
                            signatures.append(infer_signature(X_test, y_pred))
                            
                            #mlflow.evaluate(data=X_test, model_type="lstm")
                            
                            #mse = mean_squared_error(y_test, y_pred)

                        mlflow.log_metric("cross_validation_mse", np.mean(cross_validation_mses))
                        mlflow.log_metric("cross_validation_rmse", np.mean(cross_validation_rmses))
                        mlflow.log_metric("cross_validation_mapes", np.mean(cross_validation_mapes))
                        mlflow.keras.log_model(model, "model", registered_model_name="lstm_stock_predictions", signature=signatures[0])
                        mlflow.log_artifact(temp_artifact_dir)
                        mlflow.end_run()