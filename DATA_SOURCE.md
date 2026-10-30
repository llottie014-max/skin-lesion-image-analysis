# 数据来源与使用说明

本项目使用真实公开的 **DermaMNIST 64×64** 数据，属于 MedMNIST+，源自 HAM10000 皮肤镜图像数据集。main.py 完成基础统计与图像处理；可选 train_baseline.py 使用逻辑回归与随机森林开展原官方分区的教学分类对照，不构成疾病诊断。

## 下载与核验

- 官方元数据：[MedMNIST 的 info.py](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
- 数据文件：[dermamnist_64.npz（Zenodo 官方分发）](https://zenodo.org/records/10519652/files/dermamnist_64.npz?download=1)
- 文件 MD5：`b70a2f5635c6199aeaa28c31d7202e1f`
- 官方使用说明：[README](https://github.com/MedMNIST/MedMNIST#data-files)
- 64 尺寸与原版保持相同划分和样本索引：[MedMNIST+ 说明](https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md)

首次运行由程序下载并校验文件。文件已下载且完整时，可以用 `.venv/bin/python main.py --offline` 离线复跑。MD5 用于核对下载文件是否与官方发布版本一致。

## 数据结构与标签

数据共 10015 张 RGB 图像，每张为 64×64 像素。官方划分是 train 7007 张、val 1003 张、test 2005 张，约为 7:1:2。本项目默认分析 **train 的 7007 张图像**；图表和占比的分母以实际选择的划分为准。

NPZ 包含 `train_images`、`train_labels`、`val_images`、`val_labels`、`test_images`、`test_labels` 六个数组。图像形状为 `(N, 64, 64, 3)`；标签形状为 `(N, 1)`，编号含义如下：

| 编号 | 官方英文名称 | 中文说明 |
| --- | --- | --- |
| 0 | actinic keratoses and intraepithelial carcinoma | 日光性角化病及上皮内癌 |
| 1 | basal cell carcinoma | 基底细胞癌 |
| 2 | benign keratosis-like lesions | 良性角化样病变 |
| 3 | dermatofibroma | 皮肤纤维瘤 |
| 4 | melanoma | 黑色素瘤 |
| 5 | melanocytic nevi | 黑色素细胞痣 |
| 6 | vascular lesions | 血管性病变 |

这些是数据集给定的类别标签，并非本程序预测的结果。不要按英文名称排序后重新解释原始编号。

## 许可

DermaMNIST 数据采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)：使用时署名、保留许可信息、说明修改，并遵守非商业限制。MedMNIST 仓库代码的 Apache 2.0 许可与数据许可不同。[官方许可说明](https://github.com/MedMNIST/MedMNIST#license-and-citation)

报告中的处理后图像应注明“DermaMNIST/HAM10000，CC BY-NC 4.0；经灰度化等处理”。官方明确数据不用于临床使用，本课程结果也不构成诊断依据。

## 统计解释与划分局限

类别比例表示这个数据集所收集的**图像组成**，不是患者患病率、一般人群患病率或临床就诊概率。一位患者或同一病灶可能对应多张照片；公开 NPZ 中的图像条数不能直接当作独立患者数。

2025 年的数据质量研究指出，原 DermaMNIST 的同一病灶照片可能跨训练、验证和测试集：7470 个独立病灶中有 1006 个（约 13.47%）跨越多个划分。因此不能把这些划分声称为独立病灶或独立患者划分。64 尺寸沿用原始划分，也继承这一局限。见 [研究论文](https://www.nature.com/articles/s41597-025-04382-5) 与 [作者项目页](https://derm.cs.sfu.ca/critique/)。

基础分析保留官方划分用于描述性统计。可选逻辑回归实验也保留官方划分，所有成绩明确限定为有上述局限的教学benchmark；精确像素SHA256审计不会改变划分或标签，也不能排除同病灶不同照片。正式研究仍需核查病灶身份及近重复图像，采用合适分组划分，不能仅凭当前 test 集宣称临床泛化能力。低分辨率和拍摄条件也会影响细节与颜色统计。

## 参考文献

官方要求引用 MedMNIST 论文与所用子集的源数据论文，以下条目来自 [官方 BibTeX](https://medmnist.com/assets/v2/medmnist.bib)：

1. Yang, J., Shi, R., Wei, D., et al. MedMNIST v2—A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10, 41 (2023). [DOI](https://doi.org/10.1038/s41597-022-01721-8).
2. Yang, J., Shi, R., & Ni, B. MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis. *ISBI*, 191–195 (2021).
3. Tschandl, P., Rosendahl, C., & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 180161 (2018). [DOI](https://doi.org/10.1038/sdata.2018.161).
4. Codella, N., Rotemberg, V., Tschandl, P., et al. Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC). arXiv:1902.03368 (2019). [论文](https://arxiv.org/abs/1902.03368).
5. Abhishek, K., Jain, A., & Hamarneh, G. Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets. *Scientific Data*, 12, 196 (2025). [DOI](https://doi.org/10.1038/s41597-025-04382-5).
