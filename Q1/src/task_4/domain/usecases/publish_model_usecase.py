import os
import subprocess
import sys
from core.util.logger import ILogger

class PublishModelUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def execute(self, model_path: str) -> bool:
        self.logger.info("Executing PublishModelUseCase...")
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}!")
            return False

        # Find workspace root (5 levels up from task_4/domain/usecases)
        usecases_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.abspath(os.path.join(usecases_dir, "../../../../../"))
        self.logger.info(f"Workspace root resolved to: {workspace_root}")

        # Resolve DVC executable
        python_bin_dir = os.path.dirname(sys.executable)
        dvc_path = os.path.join(python_bin_dir, "dvc")
        
        if not os.path.exists(dvc_path):
            self.logger.warning(f"DVC executable not found in python bin directory ({dvc_path}). Falling back to 'dvc'.")
            dvc_path = "dvc"
        else:
            self.logger.info(f"Using DVC executable: {dvc_path}")

        # Calculate relative path of model from workspace root
        rel_model_path = os.path.relpath(model_path, workspace_root)
        self.logger.info(f"Relative model path from workspace root: {rel_model_path}")

        # 1. Run DVC add
        self.logger.info(f"Running: {dvc_path} add {rel_model_path}")
        add_result = subprocess.run(
            [dvc_path, "add", rel_model_path],
            cwd=workspace_root,
            capture_output=True,
            text=True
        )
        if add_result.returncode != 0:
            self.logger.error(f"Failed to add model to DVC:\nSTDOUT: {add_result.stdout}\nSTDERR: {add_result.stderr}")
            return False
        self.logger.info(f"Successfully added model to DVC:\n{add_result.stdout}")

        # 2. Run DVC push
        self.logger.info(f"Running: {dvc_path} push {rel_model_path}.dvc")
        push_result = subprocess.run(
            [dvc_path, "push", f"{rel_model_path}.dvc"],
            cwd=workspace_root,
            capture_output=True,
            text=True
        )
        if push_result.returncode != 0:
            self.logger.error(f"Failed to push model to DVC remote storage:\nSTDOUT: {push_result.stdout}\nSTDERR: {push_result.stderr}")
            return False
        self.logger.info(f"Successfully pushed model to DVC remote storage:\n{push_result.stdout}")

        self.logger.info("PublishModelUseCase execution completed successfully.")
        return True
