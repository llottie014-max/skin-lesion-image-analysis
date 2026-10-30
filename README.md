# Skin Lesion Image Analysis

用 Python 分析 DermaMNIST 皮肤镜图像，提取24维颜色和灰度特征，对比逻辑回归与随机森林。

## 结果

| 模型 | 测试集准确率 | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| 多数类基线 | 66.88% | 0.143 | 0.115 |
| 逻辑回归 | 52.32% | 0.486 | 0.361 |
| 随机森林 | 70.37% | 0.260 | 0.297 |

随机森林准确率为70.37%，比逻辑回归高18.05个百分点，比多数类基线高3.49个百分点；macro-F1低于逻辑回归。两种分类器的验证集选参目标不同，具体设置见[实验结果](实验结果.md)。

![三种模型在相同特征上的测试集成绩](assets/model_comparison.svg)

数据使用原官方划分：train 7007张、val 1003张、test 2005张。[数据来源与划分说明](数据说明.md) · [完整成绩CSV](assets/results.csv)

## 安装

使用 Python 3.12：

```bash
git clone https://github.com/llottie014-max/skin-lesion-image-analysis.git
cd skin-lesion-image-analysis
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-ml-lock.txt
```

Windows 用 `py -3.12 -m venv .venv` 创建环境，之后将 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`。

也可运行 `python3 bootstrap.py --ml` 自动配置已有的Python 3.12。macOS支持双击 `配置环境.command` 和 `运行项目.command` 运行基础分析。

## 运行

```bash
# 基础图像统计：首次运行自动下载数据，约100MB
.venv/bin/python main.py

# 缓存数据后，可离线指定分区和示例图像
.venv/bin/python main.py --offline --split val --image-index 10

# 训练多数类基线、逻辑回归和随机森林，输出验证集结果
.venv/bin/python train_baseline.py --offline --random-forest

# 使用选定模型输出测试集结果
.venv/bin/python train_baseline.py --offline --random-forest --evaluate-test
```

不加 `--random-forest` 时只运行多数类基线和逻辑回归。模型只在train拟合，参数由val选择；默认不输出test成绩。

## 文件

| 文件 | 用途 |
|---|---|
| `main.py`、`workflow.py` | 数据下载、读取、统计与图像处理 |
| `train_baseline.py`、`experiment.py` | 分类实验入口与运行流程 |
| `features.py` | 24维特征提取 |
| `baseline.py` | 分类器、验证集选参与指标计算 |
| `baseline_outputs.py` | 预测表、混淆矩阵与对比图 |
| `quality.py` | 精确像素重复与标签冲突检查 |
| `bootstrap.py` | 环境配置与依赖检查 |
| `requirements*.txt` | 基础与分类实验依赖、锁定版本 |
| `tests/`、`tests_ml/` | 基础流程与分类实验测试 |
| `assets/` | 本次实验的汇总成绩与展示图 |
| [数据说明.md](数据说明.md) | 下载地址、标签、许可和引用 |
| [实验结果.md](实验结果.md) | 特征定义、参数与评估结果 |

## 输出

基础分析写入 `outputs/train/` 或所选分区目录，包括类别统计、逐图特征、样本图、处理对比图和直方图。

分类实验写入 `outputs/baseline/`，开启森林后写入 `outputs/random_forest/`。验证与测试结果分别保存在 `validation/` 和 `test_evaluation/`：

- `results.md`、`experiment.json`：成绩、参数、版本和运行记录。
- `*_predictions.csv`、`*_classification.csv`：逐图预测与逐类指标。
- `*_confusion.csv`、`*_confusion_matrix.png`：混淆矩阵。
- `*_model_comparison.png`：模型对比图。
- `random_forest_candidates.csv`、`random_forest_feature_importance.csv`：森林候选参数与特征重要性。
- `data_quality.json`：像素重复和标签冲突记录。

完整数据、运行输出、日志和虚拟环境保存在本地；仓库只保留汇总结果。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s tests_ml
.venv/bin/python -m pip check
```

数据来自 [MedMNIST / HAM10000](https://medmnist.com/)，采用CC BY-NC 4.0许可，用于非临床图像分类实验。
