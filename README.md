# KG-LCD-Net 代码与数据快照

本仓库保存 KG-LCD-Net 眼底血管分割论文的研究代码、当前已有的 DRIVE 实验缓存，以及按 CC BY 4.0 许可再分发的 CHASE_DB1 原始数据。`code/lcd.py` 同时包含端点桥接和独立的多层超像素图分支；**当前已报告的 DRIVE 桥接结果只调用 `lcd_reconnect`，没有调用多层超像素图分支**。

## 内容

- `code/`：模型、损失、数据读取、训练、推理、桥接和部分绘图脚本。
- `data/cache/`：已有的 DRIVE 模型权重、概率图、训练历史及汇总 JSON。这些文件是现存实验资产的快照；请结合论文中的实验设置阅读，不能把它们视为独立重新运行的验证结果。
- `data/CHASEDB1.zip`：Kingston University 发布的 CHASE_DB1 数据集；来源、作者与许可见 [`DATA_SOURCES.md`](DATA_SOURCES.md)。

目前没有可核查的 CHASE_DB1 完整模型结果或 CHASE 桥接结果，因此本仓库不提供这两类指标。`code/run_experiments.py` 包含计划的训练与评估流程，但重新训练需要显著计算时间；本次公开快照没有重新训练。`lcd_sensitivity.json` 是已有数值记录，当前没有独立的参数扫描生成脚本。部分表格和概念图的全部生成链也不在本仓库中。

## 数据目录

DRIVE 原始图像和标注须从[官方项目页面](https://drive.grand-challenge.org/)取得。本仓库没有再分发 DRIVE 原始数据，也没有为其另行授予使用许可。整理为：

```text
data/DRIVE/training/images/
data/DRIVE/training/1st_manual/
data/DRIVE/training/mask/
data/DRIVE/test/images/
data/DRIVE/test/1st_manual/
data/DRIVE/test/mask/
```

CHASE_DB1 解压到 `data/CHASEDB1/`，其中直接包含 `Image_*.jpg` 和 `Image_*_1stHO.png`。论文使用前 20 张图像训练、后 8 张测试；这属于论文指定划分，不是数据集官方划分。

## 环境与运行入口

建议使用隔离的 Python 环境，安装 `requirements.txt`。依赖版本未锁定；发布时仅对代码做了语法检查，未在全新环境中完成端到端复跑。

```bash
python -m pip install -r requirements.txt
python code/dataset_prep.py
python code/run_experiments.py
python code/make_figures_drive.py
```

最后两条命令会运行训练或生成图件，应在确认数据布局、计算预算和所需输出后分别执行。已有 DRIVE 数值见 `data/cache/results_drive.json`，参数记录见 `data/cache/lcd_sensitivity.json`。`code/make_figures_drive.py` 还需要 DRIVE 原始图像和仓库中的概率图缓存；它不生成论文中的每一张图。

## 发布副本与原始代码的区别

本目录从论文所属文件夹整理而来。为了让路径可迁移，发布副本仅把历史脚本中的 `/mnt/agents/output/...` 和 `/tmp/chase_cache` 改为以仓库根目录为基准的路径；模型、损失和桥接算法没有因打包而修改。两个 `drive_backbone*_probs.npz` 文件做了无损 ZIP 重压缩，数组名称、形状、类型和逐元素数值均保持一致。源文件与发布副本的 SHA-256 记录在 `RELEASE_MANIFEST.tsv`。没有将历史备份、重复缓存、未完成 CHASE 中间数组或其他论文的文件纳入仓库。

公开仓库中的代码未另附开源许可证；数据集许可只适用于对应数据。使用或再分发 CHASE_DB1 时请保留其归属信息并遵守 CC BY 4.0。
