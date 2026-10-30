# Skin Lesion Image Analysis

基于 Python 的皮肤病变图像分析与可视化学习项目。

A beginner-friendly Python project for DermaMNIST image analysis, with an optional reproducible classical machine learning baseline. No deep learning or clinical diagnosis.

## 功能

- 自动下载 DermaMNIST 64×64 并核对官方 MD5。
- 验证图像尺寸、类型、标签和数据分区。
- 统计七类数量与占比，导出 CSV。
- 提取逐图 RGB 均值、灰度均值和标准差。
- 对指定图像进行缩放、灰度化和直方图分析。
- 保存 300 dpi 图表和 JSON 运行摘要。

项目通过 Python 脚本运行，结果保存为文件。

新增进阶实验：24维可解释特征、精确像素重复审计、多数类对照与逻辑回归分类，按验证集选择参数。真实实验与局限见 [实验结果.md](实验结果.md)。

随机森林改进：在相同特征和分区上，以验证集准确率选择随机森林参数，与两种原基线比较；结果与准确率/macro-F1取舍见 [随机森林.md](随机森林.md)。

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

## 进阶：基础分类实验

原来的 main.py 保持独立；需要分类实验时再安装可选依赖：

```bash
.venv/bin/python -m pip install -r requirements-ml-lock.txt
# 也可使用 python3 bootstrap.py --ml 自动配置
.venv/bin/python train_baseline.py --offline
```

默认输出 `outputs/baseline/validation/`，包含数据质量审计、候选参数分数、两种方法的验证集预测与指标、混淆矩阵、错误示例和完整配置。默认不计算测试集预测或指标；但会读取三个分区做精确重复和标签冲突审计。

确定特征和选参协议后，可以显式开启原官方测试集评估：

```bash
.venv/bin/python train_baseline.py --offline --evaluate-test
```

输出到独立的 `outputs/baseline/test_evaluation/`。缩放器和模型只在 train 拟合，用 val 的 macro-F1 在 C=0.1/1/10 中选参数；不合并 train+val 重训，不根据 test 再调参。原数据有已报告的同病灶跨分区局限，所有指标仅用于原官方分区的教学对照。

### 随机森林：以提高准确率为目标

```bash
.venv/bin/python train_baseline.py --offline --random-forest
# 确定参数后再显式评估原官方test：
.venv/bin/python train_baseline.py --offline --random-forest --evaluate-test
```

无需增加依赖。结果分别保存到 `outputs/random_forest/validation/` 和 `outputs/random_forest/test_evaluation/`。随机森林不需要标准化，仍使用相同24维特征；300棵树、固定种子42、两个CPU线程，在8个预先固定候选中按val accuracy选择参数。LR原来的macro-F1选择目标不变，因此不能把两种模型说成优化同一指标。

新增三模型对比图、RF预测/分类报告/混淆矩阵、验证集错误例、完整候选和不纯度特征重要性。重要性不代表因果解释。测试集在前轮已经查看，本轮是同分区的探索性比较，不是全新独立验证。

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


## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

测试使用小型人工数组，不自动下载数据；完整分析通过 main.py 在真实数据上运行。

安装可选机器学习依赖后，再运行 `.venv/bin/python -m unittest discover -s tests_ml -v`。基础测试与可选实验测试分开，原分析项目不强制安装 scikit-learn。

## 来源与局限

使用 [MedMNIST / DermaMNIST](https://github.com/MedMNIST/MedMNIST)，源数据为 HAM10000。数据许可为 CC BY-NC 4.0，引用见 [数据说明.md](数据说明.md)。仓库不包含原始数据或派生图片。

基础统计展示已有标签；分类实验将真实标签与模型预测分开保存。图像比例不是患者患病率。低分辨率、拍摄条件及已报告的同病灶跨划分问题限制模型结果的解释。

当前已完成图像分析、逻辑回归与随机森林教学对照，尚未训练深度学习模型。进一步研究前需取得病灶身份映射并使用合理的分组划分；像素哈希检查不能替代这一工作。

代码尚未指定开源许可证，数据许可与代码权利分开处理。
