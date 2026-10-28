# Skin Lesion Image Analysis

基于 Python 的皮肤病变图像分析与可视化学习项目。

A beginner-friendly Python project for descriptive analysis and visualization of DermaMNIST skin lesion images. No classifier is trained in the current version.

## 功能

- 自动下载 DermaMNIST 64×64 并核对官方 MD5。
- 验证图像尺寸、类型、标签和数据分区。
- 统计七类数量与占比，导出 CSV。
- 提取逐图 RGB 均值、灰度均值和标准差。
- 对指定图像进行缩放、灰度化和直方图分析。
- 保存 300 dpi 图表和 JSON 运行摘要。

项目通过 Python 脚本运行，结果保存为文件。

## 快速开始

推荐 Python 3.12：

```bash
git clone https://github.com/llottie014-max/skin-lesion-image-analysis.git
cd skin-lesion-image-analysis
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python main.py
```

首次运行下载约 100 MB，需要网络。数据缓存后可离线运行：

```bash
.venv/bin/python main.py --offline
.venv/bin/python main.py --offline --image-index 100
.venv/bin/python main.py --offline --split val --image-index 10
```

Windows 使用 `py -3.12 -m venv .venv`，之后用 `.venv\Scripts\python.exe` 代替 `.venv/bin/python`。

### 可选自动配置

```bash
python3 bootstrap.py --check  # 只检查
python3 bootstrap.py          # 寻找已安装的3.12，配置项目环境
```

macOS 也可双击 `配置环境.command` 和 `运行项目.command`；运行脚本会检查环境、保存终端日志并打开结果文件夹。

配置工具不下载解释器或修改系统 Python。环境损坏时要求先改名保存；安装依赖需要网络。锁文件记录已测试版本，`requirements.txt` 列出允许的范围。

## 数据与输出

| 分区 | 图像数 | 单张图像形状 |
|---|---:|---|
| train | 7,007 | 64×64×3 RGB |
| val | 1,003 | 64×64×3 RGB |
| test | 2,005 | 64×64×3 RGB |

默认结果在 `outputs/train/`，各分区单独保存。

| 文件 | 内容 |
|---|---|
| `class_statistics.csv` | 七类数量及占比 |
| `image_features.csv` | 逐图标签和基础像素特征 |
| `class_distribution.png` | 类别统计图 |
| `sample_gallery.png` | 每类最多三个样本，固定随机种子 |
| `processing_comparison.png` | 同一图像的原图、缩放、灰度对比 |
| `intensity_histogram.png` | 同一图像的 RGB 与灰度直方图 |
| `first_image.png` | 当前分区 index=0 原图 |
| `example_image.png` | `--image-index` 指定的原图 |
| `summary.json` | 来源、分区、数量和示例编号 |

`--image-index` 只改变示例处理，不改变全分区统计。编号从0开始；重复运行更新同分区文件。数据、输出、日志与虚拟环境不上传。

训练集复现计数（标签0～6）：228、359、769、80、779、4693、99。

## 学习路线

1. 阅读 [分步学习指南](LEARNING.md)，理解图像数组、统计和文件输出。
2. 阅读 `main.py` 的三个运行选项。
3. 阅读 `workflow.py` 的 `load_dataset`、`class_statistics`、`to_gray`，最后看 `run_analysis`。
4. 按 [演示与练习](DEMO_WALKTHROUGH.md) 核对结果并独立修改。

`bootstrap.py` 是环境辅助工具，可以稍后再读。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

测试使用小型人工数组，不自动下载数据；完整分析通过 main.py 在真实数据上运行。

## 来源与局限

使用 [MedMNIST / DermaMNIST](https://github.com/MedMNIST/MedMNIST)，源数据为 HAM10000。数据许可为 **CC BY-NC 4.0**，引用见 [DATA_SOURCE.md](DATA_SOURCE.md)。仓库不包含原始数据或派生图片。

类别是已有标签，不是程序预测；图像比例不是患者患病率。低分辨率、拍摄条件及已报告的同病灶跨划分问题限制后续建模解释。

当前仅完成描述性统计和图像处理，尚未训练深度学习模型。后续先核查病灶层面划分，再建立分类基线和开展对比实验。

项目在 AI 工具辅助下整理，面向边做边学。请真实运行、调试和独立修改，并遵守课程工具使用与引用要求。代码尚未指定开源许可证，数据许可与代码权利分开处理。
