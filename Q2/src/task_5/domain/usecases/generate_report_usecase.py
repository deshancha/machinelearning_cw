import json
import os
import numpy as np
from core.util.logger import ILogger

class GenerateReportUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def _convert_to_serializable(self, val):
        if isinstance(val, (np.integer, np.int64)):
            return int(val)
        elif isinstance(val, (np.floating, np.float64)):
            return float(val)
        elif isinstance(val, (bool, np.bool_)):
            return bool(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        else:
            return val

    def execute(self, results_data: dict, output_json_path: str = "results.json") -> None:
       
        serializable_results = self._serialize_dict(results_data)
        
        with open(output_json_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        self.logger.info("Pipeline results saved successfully.")

    def _serialize_dict(self, d: dict) -> dict:
        serialized = {}
        for k, v in d.items():
            if isinstance(v, dict):
                serialized[k] = self._serialize_dict(v)
            elif isinstance(v, list):
                serialized[k] = [self._convert_to_serializable(x) for x in v]
            else:
                serialized[k] = self._convert_to_serializable(v)
        return serialized
