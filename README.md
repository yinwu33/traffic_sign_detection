# Traffic Sign Detection Fine-tuning Pipeline

该仓库提供了一个最小化的流水线，用于基于 Hugging Face 上的公开数据集和您自己的数据集，微调 Ultralytics YOLOv8 目标检测模型。您只需要实现自定义的数据集构建器，其余步骤由脚本自动完成。

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 如果需要 GPU 训练，请确保已经安装了与 CUDA 对应版本的 PyTorch，然后再安装 `ultralytics`。

## 配置文件

在 `configs/sample_train.yaml` 中提供了一个模板，关键字段说明如下：

- `working_dir`：所有中间产物（YOLO 数据集、`data.yaml`、训练结果）会写入此目录。
- `huggingface`：定义要下载的数据集、使用的 split、缓存目录等。
- `local_dataset`：指向您稍后编写的数据集构建器（模块名、类名、初始化参数）。
- `yolo`：训练超参数，直接传递给 `YOLO.train`。

复制模板并根据需要修改，例如：

```bash
cp configs/sample_train.yaml configs/train.yaml
# 然后编辑 configs/train.yaml
```

## 编写自定义数据集构建器

您需要创建一个继承自 `tsd.datasets.BaseDatasetBuilder` 的类。例如在项目根目录下的 `dataset.py` 中：

```python
from pathlib import Path
from tsd.datasets import BaseDatasetBuilder, DatasetArtifacts, DatasetSplit


class CustomDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, root: str):
        self.root = Path(root)

    def prepare(self, work_dir: Path, force_rebuild: bool = False) -> DatasetArtifacts:
        dataset_root = work_dir / "local_custom"
        dataset_root.mkdir(parents=True, exist_ok=True)

        # 假设您已经将数据整理为 YOLO 目录结构
        splits = {
            "train": DatasetSplit(
                images=self.root / "train/images",
                labels=self.root / "train/labels",
            ),
            "val": DatasetSplit(
                images=self.root / "val/images",
                labels=self.root / "val/labels",
            ),
        }

        class_names = ["speed_limit", "stop", "yield"]
        return DatasetArtifacts(splits=splits, class_names=class_names, dataset_name="local")
```

确保 `class_names` 的顺序与标签文件中使用的类别 id 完全一致，并且与 Hugging Face 数据集的类别顺序相同。

## 运行流程

```bash
python -m tsd.train --config configs/train.yaml
```

常用参数：

- `--prepare-only`：仅构建数据集，不启动训练。
- `--no-hf`：跳过 Hugging Face 数据，仅使用本地数据。
- `--force-rebuild`：强制重新生成 YOLO 数据目录。

训练脚本会：

1. 下载并转换 Hugging Face 数据集为 YOLO 所需的 `images/labels` 结构。
2. 调用您的数据集构建器并合并数据。
3. 生成 `data.yaml`。
4. 调用 Ultralytics YOLO 执行训练和（可选的）验证。

## Hugging Face 数据集要求

当前转换器针对常见的目标检测模式：样本包含 `image`（PIL Image 或 NumPy 数组）与 `objects` 字段，且其中有 `bbox`（左上角 + 宽高）和 `category`（类别 id 或名称）。若您的数据集字段名称不同，可在配置文件中覆盖 `image_key`、`annotations_key`、`bbox_key`、`category_key`。

若无法自动推断类别名称，请在配置中提供 `class_names`。

## 下一步

- 根据需要编写 `dataset.py`，实现自定义数据集构建逻辑。
- 运行 `python -m tsd.train --config <your-config>` 准备并训练模型。
- 如需持续训练或调参，在配置文件的 `yolo` 块中更新相应超参数。
