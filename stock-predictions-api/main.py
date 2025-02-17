from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import os
import pandas as pd
from configuration.logger import get_logger
from configuration.environment import get_config
from jobs import scheduler
from services.data import *
from services.model import ModelService
from api.formatter import format_to_response_data

logger = get_logger(__name__)
config = get_config()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

# Instancia os serviços que serão utilizados para acessar os dados e as predições
price_service = PriceService()
model_service = ModelService(model_name=config.model.name, model_version=config.model.version, mlflow_tracking_uri=config.model.mlflow_tracking_uri)

# Inicia o agendador de jobs que executará as tarefas de import dos dados de forma incremental
# e também a execução das predções.
scheduler.start()

def warmup():
    try:
        data = price_service.get_history(days=1825)
        scaled_data = model_service.scaler.transform(data[['close_price']].values)
        return data, scaled_data
    except Exception as e:
        logger.error(e)
        raise Exception("Erro ao fazer o warmup do modelo ou dados")

# Inicia o modelo e carrega os dados das ações
data, scaled_data = warmup()
reverse_predictions = model_service.roll_forward_prediction(scaled_data, config.model.max_preditions)

# Inicia o framework FastAPI
app = FastAPI(title="Stock Predictions API")
Instrumentator().instrument(app).expose(app)


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
    return format_to_response_data(data, reverse_predictions, model_service.model_version, prediction_days, history_days)
