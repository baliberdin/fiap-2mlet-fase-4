from .scheduler import AbstractJob
from configuration.environment import JobConfig, AppConfig, ModelConfig
import pandas as pd
import yfinance as yf
from database.repository import *
from database.stock_db import *
import datetime
from configuration.logger import get_logger
from services.model import ModelService
from services.data import PriceService

logger = get_logger('stock_api')


class StockPriceHistoryJob(AbstractJob):
    """
    Classe que implementa o Job que atualiza os dados de preços das ações no
    banco de dados em um intervalo de tempo.
    """
    def __init__(self, config: JobConfig):
        super().__init__(config)
        self.price_service = PriceService()
        self.price_history_repository = GenericRepository(engine=get_engine(), table_entity=PriceHistory)
        self.prediction_history_repository = GenericRepository(engine=get_engine(), table_entity=PredictionPriceHistory)
        self.model_config = ModelConfig(**config.params['model'])
        self.model_service = ModelService(self.model_config.name, self.model_config.version, self.model_config.mlflow_tracking_uri)
        
    def run(self):
        logger.info(f"Running [{datetime.datetime.now()}]: {self.config.name}")
        last_result = self.price_history_repository.get_last_rows(sort_column='reference_date', 
                                                                 sort_direction='desc', limit=1)
        
        if len(last_result) < 1:
            # Em caso da base estar vazia: carrega um histórico de ~5 anos
            initial_date = datetime.date.today() - datetime.timedelta(days=1825)
            self.fill_last_days(initial_date)
        else:
            # Em caso de já existir dados na base: carrega os dias restantes
            initial_date = last_result[0]['reference_date'] + datetime.timedelta(days=1)
            if (datetime.date.today() - datetime.timedelta(days=1)) == initial_date:
                logger.info("Os dados mais recentes já estão carregados na base.")
            else:
                logger.info(f"Carregando dados das ações a partir de: {initial_date}")
                self.fill_last_days(initial_date)
                
        # Inicia o processo de preencher os dados das previsões
        last_predition_result = self.prediction_history_repository.get_last_rows(sort_column='reference_date', sort_direction='desc', limit=1)
        
        if len(last_predition_result) < 1:
            # Gera previsões para toda a base de preço
            data = self.price_service.get_history(days=1825)
            
            for indx in range(0, (len(data.index) - self.model_service.loopback)):
                reference_date = data[['reference_date']].values[indx+self.model_service.loopback]
                slice = data[['close_price']].values[indx:indx+self.model_service.loopback]
            
                scaled_data = self.model_service.scaler.transform(slice)
                reverse_predictions = self.model_service.roll_forward_prediction(scaled_data, 1)
                
                register = pd.DataFrame({"ticker": self.config.params['ticker'], "reference_date": reference_date, "predicted_price": reverse_predictions[-1], "model_version":self.model_service.model_version.version})
                register.to_sql(name=PredictionPriceHistory.__tablename__, if_exists='append', 
                        index=False, con=get_connection(), index_label='id')
        else:
            # Gera previsões apenas para as datas faltantes
            data = self.price_service.get_history(days=self.model_service.loopback*2)
            
            for indx in range(0, (len(data.index) - self.model_service.loopback)):
                reference_date = data[['reference_date']].values[indx+self.model_service.loopback]
                
                if last_predition_result[0]['reference_date'] < reference_date:
                    slice = data[['close_price']].values[indx:indx+self.model_service.loopback]
                
                    scaled_data = self.model_service.scaler.transform(slice)
                    reverse_predictions = self.model_service.roll_forward_prediction(scaled_data, 1)
                    
                    register = pd.DataFrame({"ticker": self.config.params['ticker'], "reference_date": reference_date, "predicted_price": reverse_predictions[-1], "model_version":self.model_service.model_version.version})
                    register.to_sql(name=PredictionPriceHistory.__tablename__, if_exists='append', 
                            index=False, con=get_connection(), index_label='id')
                else:
                    print(f"Ignoring data {reference_date}")
        
        
    def retrieve_stock_data(self, ticker, start_dt):
        """
        Função que carrega os dados históricos do Yahoo Finance
        """
        data = yf.download(ticker, start=start_dt)
        data = data[['Close']].reset_index()
        data.columns = ['reference_date','close_price']
        data['reference_date'] = data['reference_date'].apply(lambda x: x.date() )
        data['ticker'] = data['reference_date'].apply(lambda x: self.config.params['ticker'] )
        return data
    
    def fill_last_days(self, start_dt:datetime.date):
        """
        Função que salva os dados históricos dos últimos dias no banco de dados
        """
        data = self.retrieve_stock_data(self.config.params['ticker'], start_dt)
        if len(data.index) > 0:
            data = data.loc[(data['reference_date'] >= start_dt) & (data['reference_date'] < datetime.date.today())]
            data.to_sql(name=PriceHistory.__tablename__, if_exists='append', 
                        index=False, con=get_connection(), index_label='id')
        else:
            logger.info("Não existem novos dados para serem adicionados na base")