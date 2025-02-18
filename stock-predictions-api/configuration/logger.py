import logging
import logging.config

# Carrega as configurações de Logs que está no arquivo logging.conf
logging.config.fileConfig('logging.conf', disable_existing_loggers=True)


def get_logger(logger_name):
    """
    Função para ter acesso a um logger com as configurações definidas no arquivo logging.conf
    Args:
        logger_name (str): Nome do logger que se deseja ter acesso
    Returns:
        Logger (Logger): Uma instância de logger referente ao nome passado
        como parâmetro
    """
    return logging.getLogger('stock_api')