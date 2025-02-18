from database.repository import *
from database.stock_db import *
import pandas as pd

class PriceService:
    def __init__(self):
        engine = get_engine()
        self.priceRepository = GenericRepository(engine, table_entity=PriceHistory)
        self.predictionPriceRepository = GenericRepository(engine, table_entity=PredictionPriceHistory)
        
    def get_history(self, days: int = 200):
        data = self.priceRepository.get_last_rows(sort_column='reference_date', sort_direction='desc', limit=days)
        data = pd.DataFrame(data)[['reference_date','close_price']]
        # Inverte a ordem da lista histórica. Do mais antigo para o mais novo.
        data = data.iloc[::-1]
        return data
    
    def get_predicted_history(self, days: int = 200):
        data = self.predictionPriceRepository.get_last_rows(sort_column='reference_date', sort_direction='desc', limit=days)
        data = pd.DataFrame(data)[['reference_date','predicted_price']]
        # Inverte a ordem da lista histórica. Do mais antigo para o mais novo.
        data = data.iloc[::-1]
        return data
    
    