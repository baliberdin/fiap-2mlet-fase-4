import os
from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict
from pydantic_settings_yaml import YamlBaseSettings

import logging

logger = logging.getLogger(__name__)


class JobConfig(BaseModel):
    """
    Classe que define as configurações dos jobs que vâo ser executados em segundo plano
    """
    # Nome do Job
    name: str
    # Parâmetros customizados que o job possa necessitar para ser executado
    params: dict
    # Intervalo em segundos em que o job deve ser executado
    seconds_interval: int
    # Caminho do módulo onde está a classe que implementa o Job
    module: str
    # Nome da classe que implementa o Job
    clazz: str


class DatabaseConfig(BaseModel):
    """
    Classe que define a configuração de acesso ao banco de dados.
    """
    # Nome do schema ou database
    db_name: str
    # endereço IP ou Hostname do servidor de banco de dados
    host: str
    # Porta em que o banco de dados recebe conexões
    port: int
    # Usuário de conexão com o bando de dados
    username: str
    # Senha para acesso ao banco de dados
    password: str
    

class ModelConfig(BaseModel):
    """Classe que representa as configurações do modelo"""
    # Nome do modelo publicado no Mlflow
    name: str
    # Versão do modelo publicado no mlflow
    version: int
    # URL do serviço do mlflow
    mlflow_tracking_uri: str
    # Número máximo de dias para predições
    max_preditions: int
            

    
class AppConfig(BaseModel):
    """
    Classe que define as configurações da API
    """
    # Nome da aplicação. Será exibida na tela de documentação /docs
    application_name: str
    # Lista de jobs para serem agendados.
    jobs: list[JobConfig]
    # Configuração do acesso ao banco de dados
    database: DatabaseConfig
    # Configurações do modelo
    model: ModelConfig
    

class SettingsLoader(YamlBaseSettings):
    """
    Classe que carrega as configurações gerais da aplicação exceto as configurações de log
    """
    # Atributo que guarda as configurações da aplicação
    app_config: AppConfig
    # Carrega as configurações a partir do arquivo env.yaml
    model_config = SettingsConfigDict(secrets_dir='./secrets', yaml_file='./env.yaml', env_file_encoding='utf-8')


@lru_cache
def get_config() -> AppConfig:
    """ Captura todas as configurações definidas em variáveis de ambiente ou no arquivo .env
    Variáveis de ambiente tem precedência em relação às que estão no arquivo .env
    Todas as configurações são cacheadas usando lru_cache

    Returns:
        AppConfig: Todas as configurações da aplicação
    """

    settings = SettingsLoader().app_config
    logger.debug(f"Configuration: {settings}")
    return settings