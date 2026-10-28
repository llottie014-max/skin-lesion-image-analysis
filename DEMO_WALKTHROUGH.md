# 五分钟演示与学习练习

## 运行与展示

1. 运行 `.venv/bin/python main.py --offline`；首次运行去掉 --offline 下载数据。
2. 查看终端分区、数组形状和类别计数，确认默认 train 有7007张。
3. 打开 `outputs/train/class_distribution.png`，对照 `class_statistics.csv`。黑色素瘤779张，占约11.12%，这是图像比例，不是患病率。
4. 打开 `processing_comparison.png` 与 `intensity_histogram.png`，说明同一图像的不同表示及信息损失。
5. 打开 `sample_gallery.png`，解释已有标签、固定种子和分区内 index。
6. 在 `image_features.csv` 核对一张图的特征，在 `summary.json` 核对运行参数。

## 自己动手

```bash
.venv/bin/python main.py --offline --image-index 100
.venv/bin/python main.py --offline --split val --image-index 10
```

先观察为什么类别统计不变而直方图变化，再查看 outputs/val 中1003张图的结果。不同分区的同一编号不是同一张图。重复运行更新同分区文件，比较前请另存结果。

先学 main.py 的选项，再理解读取、统计、灰度函数，最后看完整流程。记录真实核对、修改和调试；尚未完成的练习不要写成已经完成。

当前不训练分类器。数据出处、许可和划分局限见 DATA_SOURCE.md。
