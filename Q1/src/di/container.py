from dependency_injector import containers, providers
from core.util.logger import LoggerFactory, ILogger
from core.data.manager.csv_loader_imp import CsvLoaderImp
from core.domain.manager.idata_loader import IDataLoader

from task_1.domain.usecases.fetch_raw_data_usecase import FetchRawDataUseCase
from task_1.domain.usecases.analyze_target_usecase import AnalyzeTargetUseCase

from task_2.domain.usecases.data_cleaning_usecase import DataCleaningUseCase
from task_2.domain.usecases.feature_engineering_usecase import FeatureEngineeringUseCase
from task_2.domain.usecases.generate_eda_plots_usecase import GenerateEdaPlotsUseCase

from task_3.domain.usecases.train_and_tune_usecase import TrainAndTuneUseCase

from task_4.domain.usecases.model_evaluation_usecase import ModelEvaluationUseCase
from task_4.domain.usecases.publish_model_usecase import PublishModelUseCase

from task_6.domain.usecases.income_prediction_usecase import IncomePredictionUseCase
from task_6.data.manager.api_server_imp import ApiServerImp

class AppContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    logger: providers.Provider[ILogger] = providers.Singleton(
        LoggerFactory.create,
        logger_type="console",
        name="ML_CW"
    )

    data_loader: providers.Provider[IDataLoader] = providers.Singleton(
        CsvLoaderImp,
        data_dir=config.data.data_dir,
        logger=logger
    )

    # Use Cases for Task 1
    fetch_raw_data_usecase = providers.Factory(
        FetchRawDataUseCase,
        data_loader=data_loader,
        logger=logger
    )

    analyze_target_usecase = providers.Factory(
        AnalyzeTargetUseCase,
        data_loader=data_loader,
        logger=logger
    )

    # Use Cases for Task 2
    data_cleaning_usecase = providers.Factory(
        DataCleaningUseCase,
        data_loader=data_loader,
        logger=logger
    )

    feature_engineering_usecase = providers.Factory(
        FeatureEngineeringUseCase,
        data_loader=data_loader,
        logger=logger
    )

    generate_eda_plots_usecase = providers.Factory(
        GenerateEdaPlotsUseCase,
        logger=logger
    )

    # Use Cases for Task 3
    train_and_tune_usecase = providers.Factory(
        TrainAndTuneUseCase,
        data_loader=data_loader,
        logger=logger
    )

    # Use Cases for Task 4
    model_evaluation_usecase = providers.Factory(
        ModelEvaluationUseCase,
        data_loader=data_loader,
        logger=logger
    )

    publish_model_usecase = providers.Factory(
        PublishModelUseCase,
        logger=logger
    )

    # Use Cases for Task 6
    income_prediction_usecase = providers.Factory(
        IncomePredictionUseCase,
        model_dir=config.models.model_dir,
        logger=logger
    )

    # API Server implementation for Task 6
    api_server = providers.Singleton(
        ApiServerImp,
        usecase=income_prediction_usecase,
        logger=logger
    )
