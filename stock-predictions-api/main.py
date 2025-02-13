from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from fastapi.exceptions import HTTPException
import mlflow
from mlflow.tracking import MlflowClient
import os
import joblib
import yfinance as yf
import datetime
import pandas as pd
import logging
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
TICKER = 'PETR4.SA'
MODEL_VERSION_NUMBER = 186
MAX_PREVISIONS = 50
MODEL_NAME = 'lstm_stock_predictions'
MLFLOW_TRACKING_URI = 'http://mlflow-server:5000'
yesterday = datetime.datetime.today()-datetime.timedelta(days=1)

def warmup():
    try:
        mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        model_version = mlflow_client.get_model_version(MODEL_NAME, MODEL_VERSION_NUMBER)
        run = mlflow_client.get_run(model_version.run_id)
        model = mlflow.keras.load_model(model_version.source)

        scaler_uri = f"runs:/{model_version.run_id}/scaler/scaler.joblib"
        scaler = joblib.load(mlflow.artifacts.download_artifacts(scaler_uri))

        data = yf.download(TICKER, 
                           start=datetime.datetime.today()-datetime.timedelta(days=1825), 
                           end=yesterday)
        data = data[['Close']]
        scaled_data = scaler.transform(data.values)
        
        return model, scaler, data, scaled_data, model_version, run
    except Exception as e:
        logging.error(e)
        raise Exception("Erro ao fazer o warmup do modelo ou dados")

def format_to_response_data(data: pd.DataFrame):
    df = pd.DataFrame()
    df['Date'] = data.index
    df['Close'] = data[['Close']].values
    np_data = df.to_numpy()
    response_data = []
    for reg in np_data:
        response_data.append({"date":reg[0], "close":reg[1]})
    return response_data


def roll_forward_prediction(model, scaled_data, loopback, depth:int = 5):
    step = 0
    while step < depth:
        inverse_pred = model.predict(scaled_data[-loopback:].reshape(1,loopback))
        scaled_data = (np.append(scaled_data, inverse_pred))
        step = step + 1
    return scaled_data[-depth:]
    
    
# Inicia o modelo e carrega os dados das ações
model, scaler, data, scaled_data, model_version, run = warmup()
response_data = format_to_response_data(data)
predictions = roll_forward_prediction(model, scaled_data, int(run.data.params["loopback"]) , MAX_PREVISIONS)
reverse_predictions = scaler.inverse_transform(pd.DataFrame(predictions).values).reshape(1,MAX_PREVISIONS)[-1].tolist()
predicted_prices = []
i = 0
for price in reverse_predictions:
    i = i+1
    predicted_prices.append({"date": yesterday+datetime.timedelta(days=i), "close": price})
    

app = FastAPI(title="Stock Predictions API")
instrumentator = Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root(prediction_days:int = 5, history_days:int = 365):
    #return {"model": {"name": model_version.name, "version":model_version.version}, 
    #        "prediction_days":days,"real_price":response_data, "predicted_prices":predicted_prices[:days]}
    return {"model": model_version, 
            "prediction_days":prediction_days,"real_price":response_data[-history_days:], "predicted_prices":predicted_prices[:prediction_days]}
