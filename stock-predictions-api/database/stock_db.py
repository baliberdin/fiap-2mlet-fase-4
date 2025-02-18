from sqlalchemy import create_engine
from configuration.environment import get_config

# Pega as configurações da aplicação
settings = get_config()
# Cria o acesso ao banco sqlite3 com um pool de 5 conexões
engine = create_engine(f"mysql+pymysql://{settings.database.username}:{settings.database.password}@{settings.database.host}:{settings.database.port}/{settings.database.db_name}", pool_size=5, echo=False)


def get_engine():
    """Função que rotarna a engine criada acima"""
    return engine


def get_connection():
    """Função que expõe uma forma direta de acesso a uma conexão do banco"""
    return engine.connect()

