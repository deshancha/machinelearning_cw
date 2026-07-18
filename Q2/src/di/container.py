from dependency_injector import containers, providers
from core.util.logger import LoggerFactory, ILogger
from core.data.manager.yahoo_finance_loader_imp import YahooFinanceLoaderImp
from core.domain.manager.idata_loader import IDataLoader

from core.data.manager.yahoo_finance_client_imp import YahooFinanceClientImp
from core.domain.usecases.collect_market_data_usecase import CollectMarketDataUseCase

from task_2.domain.usecases.eda_plots_usecase import EdaPlotsUseCase
from task_2.domain.usecases.stationarity_test_usecase import StationarityTestUseCase
from task_2.domain.usecases.feature_engineering_usecase import FeatureEngineeringUseCase
from task_2.domain.usecases.data_cleaning_usecase import DataCleaningUseCase

from task_3.domain.usecases.train_arima_usecase import TrainArimaUseCase
from task_3.domain.usecases.train_xgb_usecase import TrainXgbUseCase

from task_4.domain.usecases.garch_modeling_usecase import GarchModelingUseCase

from task_5.domain.usecases.evaluate_models_usecase import EvaluateModelsUseCase
from task_5.domain.usecases.generate_report_usecase import GenerateReportUseCase

class AppContainer(containers.DeclarativeContainer):
    # Configurations
    config = providers.Configuration()

    # Shared Core Services
    logger: providers.Provider[ILogger] = providers.Singleton(
        LoggerFactory.create,
        logger_type="console",
        name="ML_CW_Q2"
    )

    market_client = providers.Singleton(
        YahooFinanceClientImp,
        logger=logger
    )

    collect_market_data_usecase = providers.Factory(
        CollectMarketDataUseCase,
        market_client=market_client
    )

    data_cleaning_usecase = providers.Factory(
        DataCleaningUseCase,
        logger=logger
    )

    data_loader: providers.Provider[IDataLoader] = providers.Singleton(
        YahooFinanceLoaderImp,
        logger=logger,
        collect_usecase=collect_market_data_usecase,
        cleaning_usecase=data_cleaning_usecase
    )

    # Use Cases for Task 2 (Exploration and Preprocessing)
    eda_plots_usecase = providers.Factory(
        EdaPlotsUseCase,
        logger=logger
    )

    stationarity_test_usecase = providers.Factory(
        StationarityTestUseCase,
        logger=logger
    )

    feature_engineering_usecase = providers.Factory(
        FeatureEngineeringUseCase,
        logger=logger
    )

    # Use Cases for Task 3 (Forecasting Models)
    train_arima_usecase = providers.Factory(
        TrainArimaUseCase,
        logger=logger
    )

    train_xgb_usecase = providers.Factory(
        TrainXgbUseCase,
        logger=logger
    )

    # Use Cases for Task 4 (Volatility Modeling)
    garch_modeling_usecase = providers.Factory(
        GarchModelingUseCase,
        logger=logger
    )

    # Use Cases for Task 5 (Evaluation & Comparison)
    evaluate_models_usecase = providers.Factory(
        EvaluateModelsUseCase,
        logger=logger
    )

    generate_report_usecase = providers.Factory(
        GenerateReportUseCase,
        logger=logger
    )
