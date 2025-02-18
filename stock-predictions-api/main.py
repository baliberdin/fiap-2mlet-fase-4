from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import os
import pandas as pd
from configuration.logger import get_logger
from configuration.environment import get_config
from jobs import scheduler
from services.data import *
from services.model import ModelService
from api.formatter import format_to_response_data

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
logger = get_logger(__name__)
# Carrega as configurações da aplicação
config = get_config()

# Instancia os serviços que serão utilizados para acessar os dados e as predições
price_service = PriceService()
model_service = ModelService(model_name=config.model.name, model_version=config.model.version, mlflow_tracking_uri=config.model.mlflow_tracking_uri)

# Inicia o agendador de jobs que executará as tarefas de import dos dados de forma incremental
# e também a execução das predções.
scheduler.start()

# Cria as métricas do modelo para o prometheus
mae_gauge = Gauge("model_mae", "Mean Absolute Error (MAE) do modelo")
mape_gauge = Gauge("model_mape", "Mean Absolute Percentage Error (MAPE) do modelo")
last_error_gauge = Gauge("model_last_error", "Erro do último dia (real - predito)")

# Inicia o framework FastAPI e adiciona o endpoint de métricas do Prometheus
app = FastAPI(title="Stock Predictions API")
instrumentator = Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
def init_monitoring():
    instrumentator.add(lambda: mae_gauge)
    instrumentator.add(lambda: mape_gauge)
    instrumentator.add(lambda: last_error_gauge)

@app.get("/")
async def root(prediction_days:int = 5, history_days:int = 365):
    """
        Função raiz da API de predição de preços das ações
        Args:
            prediction_days: int - Quantidade de dias de previsão futura que será gerado
            history_days:int - Quantidade de dias de valores reais do preço da ação
        Returns:
            dict: JSON com os dados históricos e os dados previstos do preço das ações 
    """
    # Garante que o histórico solicitado seja igual ou maior que o loopback do modelo.
    history_days = history_days if history_days > model_service.loopback else model_service.loopback
    data = price_service.get_history(days=history_days)
    predicted_data = price_service.get_predicted_history(days=history_days)
    scaled_data = model_service.scaler.transform(data[['close_price']].values)
    
    # Executa a predição de alguns dias futuros
    reverse_predictions = model_service.roll_forward_prediction(scaled_data, prediction_days)
    
    return format_to_response_data(data, predicted_data, reverse_predictions, model_service.model_version, prediction_days, history_days)
