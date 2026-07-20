from dependency_injector import containers, providers
from core.util.logger import LoggerFactory, ILogger

class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Shared Core Services
    logger: providers.Provider[ILogger] = providers.Singleton(
        LoggerFactory.create,
        logger_type="console",
        name="ML_CW_Q3"
    )
