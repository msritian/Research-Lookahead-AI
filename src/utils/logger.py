import json
import os
from datetime import datetime
from typing import Any, Dict

class ExperimentLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"experiment_{timestamp}.jsonl")
        print(f"Logging experiment to: {self.log_file}")

    def log(self, event_type: str, data: Dict[str, Any]):
        """
        Logs a single event to the JSONL file.
        """
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            return obj

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": serialize(data)
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
