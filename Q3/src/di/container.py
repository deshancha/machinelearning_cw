from dependency_injector import containers, providers
from core.util.logger import LoggerFactory, ILogger
from task_2.domain.usecases.data_preparation_usecase import DataPreparationUseCase
from task_3.domain.usecases.genetic_algorithm_usecase import GeneticAlgorithmUseCase
from task_4.domain.usecases.mip_solver_usecase import MipSolverUseCase
from task_5.domain.usecases.compare_and_report_usecase import CompareAndReportUseCase

class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Shared Core Services
    logger: providers.Provider[ILogger] = providers.Singleton(
        LoggerFactory.create,
        logger_type="console",
        name="ML_CW_Q3"
    )

    data_preparation_usecase = providers.Factory(
        DataPreparationUseCase,
        logger=logger
    )

    genetic_algorithm_usecase = providers.Factory(
        GeneticAlgorithmUseCase,
        logger=logger
    )

    mip_solver_usecase = providers.Factory(
        MipSolverUseCase,
        logger=logger
    )

    compare_and_report_usecase = providers.Factory(
        CompareAndReportUseCase,
        logger=logger,
        data_prep=data_preparation_usecase,
        ga_solver=genetic_algorithm_usecase,
        mip_solver=mip_solver_usecase
    )
