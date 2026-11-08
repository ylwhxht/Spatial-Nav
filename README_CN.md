# Spatial-Nav：匿名复现代码

这是 Spatial-Nav 导航研究的匿名代码与复现说明仓库，包含训练和评估代码、
环境配置、协议检查、配置模式以及 checkpoint 校验信息。

以下大文件不放入 Git 仓库：

- 训练图像和 JSONL 数据；
- 仿真场景与受许可限制的导航网格；
- 基础模型权重；
- 训练得到的 adapter 和合并模型；
- 完整逐 episode 日志。

匿名数据集和模型制品的标识符会随审稿材料单独提供。仓库不把它们写死，
以避免泄露作者账号或非匿名镜像。

## 快速开始

在 Linux 环境中，根据 environment/ 下的配置创建训练、仿真和服务环境：

    export NAV_CACHE=/path/to/writable/cache
    export NAV_RESULTS=/path/to/results
    tools/bootstrap_envs.sh --prefix "$NAV_CACHE/envs"
    export TRAIN_PY="$NAV_CACHE/envs/train/bin/python"
    export SIM_PY="$NAV_CACHE/envs/sim/bin/python"
    export SERVE_PY="$NAV_CACHE/envs/serve/bin/python"

从审稿材料获得匿名数据集和模型制品标识符后设置：

    export SPATIAL_NAV_DATASET_ID=ANONYMOUS_ACCOUNT/ANONYMOUS_DATASET
    export SPATIAL_NAV_CHECKPOINT_REPO=ANONYMOUS_ACCOUNT/ANONYMOUS_MODELS
    export SPATIAL_NAV_DATA_ROOT="$NAV_CACHE/datasets"
    export SPATIAL_NAV_MODEL_ROOT="$NAV_CACHE/models"
    export SPATIAL_NAV_CHECKPOINT_ROOT="$NAV_CACHE/checkpoints"
    tools/download_hf_assets.sh

下载脚本要求显式提供制品标识符，不内置作者账号，并在训练前校验数据契约。

## 复现入口

- codes/scripts/train.py：分阶段训练入口；
- codes/scripts/evaluate.py：checkpoint 评估入口；
- tools/download_hf_assets.sh：下载外部制品；
- REPRODUCTION.md：环境和命令说明；
- CHECKPOINTS.md：架构、revision、路径和 checksum；
- tests/：协议和配置检查。

Qwen3-VL 复现链包含 Foundation、Move no-KD、Move OPD、adapter composition、
Stop 和 merged Stop。Move 与 Stop 使用独立的数据契约和决策接口。

## 匿名范围

仓库使用全新的 Git 历史和通用提交元数据，不包含作者姓名、私有服务器路径、
访问 token 或非匿名源账号链接。评估时需要的制品值均通过环境变量显式提供。

仿真场景和基础模型仍受其原始许可证约束，本仓库不重新分发这些资源。
