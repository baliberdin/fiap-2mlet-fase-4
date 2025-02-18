import mlflow
from mlflow.tracking import MlflowClient
import joblib
from configuration.logger import get_logger
import numpy as np
import pandas as pd


logger = get_logger(__name__)

class ModelService():
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        else:
            logger.info("ModelService has already been created")
        return cls._instance
    
    def __init__(self,model_name, model_version, mlflow_tracking_uri):
        self._instance = self
            
        self.MLFLOW_TRACKING_URI = mlflow_tracking_uri
        self.MODEL_NAME = model_name
        self.MODEL_VERSION_NUMBER = model_version
        
        logger.info("Loading ML Model...")
        self.load_model()
        self.load_scaler()
        logger.info("ML Model successfuly loaded.")
        
    
    def load_model(self):
        mlflow_client = MlflowClient(tracking_uri=self.MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)

        self.model_version = mlflow_client.get_model_version(self.MODEL_NAME, self.MODEL_VERSION_NUMBER)
        self.run = mlflow_client.get_run(self.model_version.run_id)
        self.model = mlflow.keras.load_model(self.model_version.source)
        self.loopback = int(self.run.data.params["loopback"])

        
    def load_scaler(self):
        self.scaler_uri = f"runs:/{self.model_version.run_id}/scaler/scaler.joblib"
        self.scaler = joblib.load(mlflow.artifacts.download_artifacts(self.scaler_uri))
        
    
    def roll_forward_prediction(self, scaled_data, depth:int = 5):
        step = 0
        while step < depth:
            inverse_pred = self.model.predict(scaled_data[-self.loopback:].reshape(1,self.loopback))
            scaled_data = (np.append(scaled_data, inverse_pred))
            step = step + 1
            
        predictions = scaled_data[-depth:]
        return self.scaler.inverse_transform(pd.DataFrame(predictions).values).reshape(1, depth)[-1].tolist()
    