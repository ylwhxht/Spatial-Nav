# Spatial-Nav-100K

[![模型：Anonymous](https://img.shields.io/badge/Model-Anonymous-yellow)](https://anonymous-hf.com/a/yd05u6f8p76h/) [![数据集：Anonymous](https://img.shields.io/badge/Dataset-Anonymous-yellow)](https://anonymous-hf.com/a/qw045c5e6efg/)

本仓库提供 Spatial-Nav-100K 的开源复现代码、数据校验和 checkpoint 评估入口，包含 Qwen3-VL-8B 的完整训练链，以及 Qwen3-VL-8B / InternVL3-8B 的评估链。

请从 [REPRODUCTION.md](REPRODUCTION.md) 开始：其中给出已验证的环境配置、Qwen3 训练命令、统一的 8 卡评估命令和完整验证集结果。[CHECKPOINTS.md](CHECKPOINTS.md) 记录 checkpoint 来源、版本和 SHA-256。

目录结构：`codes/` 保存训练与评估脚本，`environment/` 固定三个运行环境，`tools/` 下载并校验外部资源，`reproduction/` 保存复现日志和指标。`datasets/` 与 `checkpoints/` 由下载脚本在本地填充；大模型权重、受许可保护的场景资源和训练图像包不放入 GitHub。

基础模型权重、场景资源和训练图像包需要按复现指引单独准备，脚本会使用用户指定的本地路径。
