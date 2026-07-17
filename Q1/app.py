import os
import sys

# Usage: python app.py <task>
# t1: Fetch raw dataset & analyze target class distribution
# t2: EDA & Feature Engineering
# t3: training & hyperparameter optimization
# t4: Performance evaluation & model comparison
# t6: Start FastAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotenv import load_dotenv
from di.container import AppContainer

def get_container() -> AppContainer:
    load_dotenv()
    container = AppContainer()
    
    container.config.from_dict({
        "data": {
            "data_dir": os.path.join(os.path.dirname(__file__), "data")
        },
        "models": {
            "model_dir": os.path.join(os.path.dirname(__file__), "outputs/models")
        }
    })
    return container

def main():
    if len(sys.argv) < 2:
        print("Specify Task: python app.py <t1/t2/t3/t4/t6>")
        sys.exit(1)
        
    task = sys.argv[1].lower()
    container = get_container()
    logger = container.logger()

    if task == "t1":
        # Fetch raw
        fetch_usecase = container.fetch_raw_data_usecase()
        fetch_usecase.execute()
        
        # analyze
        analyze_usecase = container.analyze_target_usecase()
        output_dir = os.path.join(os.path.dirname(__file__), "outputs/task1")
        analyze_usecase.execute(output_dir)
        
        logger.info("Task 1 done")

    elif task == "t2":
        # Clean data
        cleaning_usecase = container.data_cleaning_usecase()
        df_clean = cleaning_usecase.execute()
        
        # Feature engineering
        fe_usecase = container.feature_engineering_usecase()
        df_processed = fe_usecase.execute(df_clean)
        
        # Generate EDA plots
        df_raw = container.data_loader().load_raw_data()
        plots_usecase = container.generate_eda_plots_usecase()
        output_dir = os.path.join(os.path.dirname(__file__), "outputs/task2")
        plots_usecase.execute(df_raw, df_processed, output_dir)
        
        logger.info("Task 2 done")

    elif task == "t3":
        # Train and tune 5 models
        train_usecase = container.train_and_tune_usecase()
        model_dir = os.path.join(os.path.dirname(__file__), "outputs/models")
        output_summary_path = os.path.join(os.path.dirname(__file__), "outputs/model_development_summary.csv")
        train_usecase.execute(model_dir, output_summary_path)
        
        logger.info("Task 3 done")

    elif task == "t4":
        # Evaluate trained models on test dataset
        eval_usecase = container.model_evaluation_usecase()
        model_dir = os.path.join(os.path.dirname(__file__), "outputs/models")
        output_dir = os.path.join(os.path.dirname(__file__), "outputs/task4")
        eval_usecase.execute(model_dir, output_dir)
        
        logger.info("Task 4 done")

    elif task == "t6":
        # Start FastAPI
        server = container.api_server()
        server.start(host="0.0.0.0", port=8000)

    else:
        logger.error(f"Task '{task}' is invalid")
        sys.exit(1)

if __name__ == "__main__":
    main()
