import traceback

from apscheduler.schedulers.background import BackgroundScheduler

from configuration.environment import get_config
import logging
from configuration.environment import JobConfig, AppConfig

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()
settings = get_config()


class AbstractJob:
    """Classe abstrata para jobs"""
    # Configuração do Job
    config: JobConfig

    def __init__(self, config: JobConfig):
        self.config = config
        pass

    def run(self):
        """Método que executa o job propriamente dito"""
        pass


def job_execution_wrapper(job_config: JobConfig):
    """Método que encpsula a execução dos jobs que serão executados pelo Scheduler
    Args:
        job_config: JobConfig - Configuração do Job
    """
    try:
        logger.info(f"Starting job {job_config.name}")

        # Extrai das configurações o módulo e a classe do job a ser executado
        mod = __import__(job_config.module, fromlist=[job_config.clazz])
        clazz = getattr(mod, job_config.clazz)
        # Instancia o Job
        job: AbstractJob = clazz(job_config)
        # Executa o Job
        job.run()
    except Exception as e:
        logger.warn(f"Error on job execution: {e}")
        traceback.print_exc()


def start():
    """Método que carrega os jobs que estão no arquivo de env e inicia o Scheduler"""

    logger.info("Starting Scheduler.")
    # Iterando sobre todos os jobs configurados no arquivo env.yaml
    for job_config in settings.jobs:
        # A primeira execução do job é disparada assim que o metodo start é chamado.
        # As execuções subsequentes obedecem a configuração `seconds_interval`
        job_execution_wrapper(job_config)
        # Adiciona o job ao scheduler baseado em um intervalo de tempo
        scheduler.add_job(job_execution_wrapper, 'interval',
                          seconds=job_config.seconds_interval, args=[job_config])
    # Inicia o scheduler
    scheduler.start()