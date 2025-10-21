import os
import math
import json
from datetime import datetime

from detectron2.engine import hooks


class EvalWriterHook(hooks.EvalHook):
    """
    Extend EvalHook to dump validation metrics to a text file every evaluation.
    """

    def __init__(self, eval_period, eval_function, output_file):
        super().__init__(eval_period, eval_function)
        self.output_file = output_file

    def _to_serializable(self, value):
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def _do_eval(self):
        results = super()._do_eval()
        if not results:
            return results

        epoch = math.ceil((self.trainer.iter + 1) / max(1, self._period))
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "iteration": self.trainer.iter + 1,
            "results": self._to_serializable(results),
        }

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        return results
