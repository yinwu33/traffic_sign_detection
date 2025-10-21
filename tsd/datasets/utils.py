import json
import random
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=None)
def load_dataset_json(json_fpath: str):
    with open(json_fpath, "r") as f:
        dataset_dicts = json.load(f)
    return dataset_dicts


def dataset_split(
    json_fpath: str,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    assert 0.0 < val_ratio < 1.0, "val_ratio must be between 0 and 1"

    json_fpath = Path(json_fpath)
    assert json_fpath.exists(), f"File {json_fpath} does not exist"
    train_out = json_fpath.parent / f"{json_fpath.stem}_train{json_fpath.suffix}"
    val_out = json_fpath.parent / f"{json_fpath.stem}_val{json_fpath.suffix}"

    random.seed(seed)

    with open(json_fpath, "r") as f:
        data = json.load(f)

    random.shuffle(data)
    n_val = max(1, int(len(data) * val_ratio))
    val = data[:n_val]
    train = data[n_val:]
    with open(train_out, "w") as f:
        json.dump(train, f)
    with open(val_out, "w") as f:
        json.dump(val, f)
