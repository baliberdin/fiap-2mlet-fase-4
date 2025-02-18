import os
import tempfile
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
from sklearn.model_selection import TimeSeriesSplit
import random
from stock_lstm_model import create_model
from data import retrieve_stock_data, format_timeseries_dataset
from utils import grid_search_build, get_env

tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)

TICKER = 'PETR4.SA'
MODEL_NAME = 'lstm_stock_predictions'
MLFLOW_TRACKING_URI = get_env('MLFLOW_TRACKING_URI', 'http://localhost:5000')

data = retrieve_stock_data(TICKER, '2018-01-01')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)
temp_dir = tempfile.TemporaryDirectory()
temp_artifact_dir = os.path.join(temp_dir.name,'scaler')
print(temp_artifact_dir)
os.mkdir(temp_artifact_dir)

# Escreve o scaler no disco para enviar posteriormente para o MLFlow como artefato.
joblib.dump(scaler, os.path.join(temp_artifact_dir,'scaler.joblib'))

data_splits = 5
tscv = TimeSeriesSplit(n_splits=data_splits)

# Conjunto de hiperparâmetros para serem testados.
params = {"batch_size": [32, 47, 75], 
          "epochs": [100, 140, 300], 
          "loopback": [10, 14, 20, 30], 
          "learning_rate": [0.0007, 0.0005, 0.0003, 0.0009],
          "units": [10, 50, 100],
          "dropout_rate": [0.2, 0.1, 0.05, 0.3]}
            

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('LSTM Stock Predictions')

param_list = grid_search_build(params)
param_list_slice = []

PAGES = 0 if 'TRAINING_SLICES' not in os.environ else int(os.environ['TRAINING_SLICES'])
PAGE = None if 'TRAINING_PAGE' not in os.environ else int(os.environ['TRAINING_PAGE'])


if PAGES > 0:
    limits = [(len(param_list)//PAGES)*PAGE, ((len(param_list) // PAGES)*PAGE) + (len(param_list) // PAGES) if PAGE < PAGES else None]
    param_list_slice = param_list[limits[0]: limits[1]]
    print(limits)
    print(f"Processo de treino dividido em partes para paralelismo. Executando página: {PAGE+1} de {PAGES}. Itens: {len(param_list_slice)}")
else:
    param_list_slice = param_list

for parameters in param_list_slice:

    with mlflow.start_run():
        mlflow.log_params(parameters)
        mlflow.keras.autolog()
        
        cross_validation_mses = []
        cross_validation_rmses = []
        cross_validation_mapes = []
        signatures = []

        # Inicia a validação cruzada gerando splits de dados para os testes
        for train_idx, test_idx in tscv.split(scaled_data):

            train_data, test_data = scaled_data[train_idx], scaled_data[test_idx]
            X_train, y_train = format_timeseries_dataset(train_data, parameters["loopback"])
            X_test, y_test = format_timeseries_dataset(test_data, parameters["loopback"])
            
            model = create_model(learning_rate=parameters["learning_rate"], units=parameters["units"], 
                                 dropout_rate=float(parameters["dropout_rate"]))
            model.fit(X_train, y_train, epochs=parameters["epochs"], batch_size=parameters["batch_size"], 
                      validation_data=(X_test, y_test), shuffle=False)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            cross_validation_mses.append(mse)
            cross_validation_rmses.append(rmse)
            cross_validation_mapes.append(mape)
            
            signatures.append(infer_signature(X_test, y_pred))
            
            
        mlflow.set_tag("type", "CrossValidation")
        mlflow.log_metric("cross_validation_mse", np.mean(cross_validation_mses))
        mlflow.log_metric("cross_validation_rmse", np.mean(cross_validation_rmses))
        mlflow.log_metric("cross_validation_mapes", np.mean(cross_validation_mapes))
        mlflow.keras.log_model(model, "model", registered_model_name=MODEL_NAME, signature=signatures[0])
        mlflow.log_artifact(temp_artifact_dir)
        mlflow.end_run()