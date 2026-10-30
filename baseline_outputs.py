"""保存分类评估表与科研风格静态图，不生成网页。"""
# 统一处理字符串和Path两种输出目录输入。
from pathlib import Path

# 生成静态评估图表。
import matplotlib.pyplot as plt
# 处理矩阵、缺失值和样本编号。
import numpy as np
# 将逐图结果和逐类指标写成表格。
import pandas as pd
# 计算七类报告和固定顺序的混淆矩阵。
from sklearn.metrics import classification_report, confusion_matrix

# 复用同一套标签和指标定义。
from baseline import LABELS, evaluate_predictions
# 类别名称与图表保存方式沿用图像分析模块。
from workflow import CLASS_NAMES, CLASS_SHORT, configure_font, save_figure


def save_evaluation(output_dir, split, model_name, truth, prediction):
    # 后面使用斜杠拼接文件名，需要先转Path。
    output_dir = Path(output_dir)
    # 先计算三项总指标，最后返回给实验摘要。
    metrics = evaluate_predictions(truth, prediction)
    # 一张图像写一行预测记录。
    pd.DataFrame({
        # 编号保持输入顺序，能回到原分区查图。
        'split': split, 'image_index': np.arange(len(truth)),
        # 真实标签、预测标签放在相邻两列。
        'true_label': truth, 'predicted_label': prediction,
        # 逐项比较，True表示这张图像预测正确。
        'correct': truth == prediction,
    # 文件名同时包含分区和模型，避免不同结果互相覆盖。
    }).to_csv(output_dir / f'{split}_{model_name}_predictions.csv', index=False, encoding='utf-8-sig')
    # 固定七类顺序，保留没有预测出来的类别。
    report = classification_report(truth, prediction, labels=LABELS,
                                   # 输出字典便于转表，无法计算的单类指标记为0。
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    # accuracy是单个总分，先从逐类字典中取出。
    accuracy = report.pop('accuracy', None)
    # 以类别名作行名，precision、recall等作列名。
    table = pd.DataFrame.from_dict(report, orient='index')
    # 报告包含accuracy时，再单独补上这一行。
    if accuracy is not None:
        # accuracy是标量，不能广播为precision/recall或误填到support。
        # 准确率行的precision、recall留空，避免误解。
        table.loc['accuracy'] = {'precision': np.nan, 'recall': np.nan,
                                 # 沿用分类报告格式，把总准确率放入f1-score列，support填总图像数。
                                 'f1-score': accuracy, 'support': len(truth)}
    # 保存每类的精确率、召回率、F1和样本数。
    table.to_csv(output_dir / f'{split}_{model_name}_classification.csv', encoding='utf-8-sig')
    # 矩阵行是真实类别，列是预测类别。
    matrix = confusion_matrix(truth, prediction, labels=LABELS)
    # 行列都使用同一套七类缩写。
    pd.DataFrame(matrix, index=CLASS_SHORT, columns=CLASS_SHORT).to_csv(
        # 保留true_class作为行名，读表时能分清方向。
        output_dir / f'{split}_{model_name}_confusion.csv', encoding='utf-8-sig', index_label='true_class')
    # 主要分类器保存矩阵图，多数类基线只导出表格。
    if model_name in {'logistic', 'random_forest'}:
        # 每一行求和得到该真实类别的图像数，保留二维形状便于广播。
        support = matrix.sum(axis=1, keepdims=True)
        # 按真实类归一化，无样本的类填0以避免除零。
        normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
        # 左图显示数量，右图显示各真实类的预测比例。
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
        # 把子图、数据、标题和是否归一化一一配对。
        for ax, values, title, normalized_flag in zip(
                # 原始数量和归一化比例使用不同的显示规则。
                axes, [matrix, normalized], ['Counts', 'Recall by true class'], [False, True]):
            # 比例色阶固定0到1，数量色阶按矩阵最大值设置。
            ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized_flag else max(1, matrix.max()))
            # 横纵坐标均按标签0到6排列。
            ax.set(xticks=LABELS, yticks=LABELS, xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT,
                   # 明确标出横轴预测、纵轴真实，避免看反。
                   xlabel='Predicted label', ylabel='True label', title=title)
            # 横轴类别名旋转45度，减少拥挤。
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            # 逐行遍历真实类别。
            for row in LABELS:
                # 逐列遍历预测类别。
                for column in LABELS:
                    # 比例显示两位小数，数量显示整数。
                    text = f'{values[row, column]:.2f}' if normalized_flag else str(values[row, column])
                    # 用色阶中点区分深浅背景。
                    threshold = 0.5 if normalized_flag else max(1, matrix.max()) / 2
                    # 将具体数值写在每个格子的中央。
                    ax.text(column, row, text, ha='center', va='center', fontsize=8,
                            # 深色背景用白字，提高可读性。
                            color='white' if values[row, column] > threshold else '#172c27')
        # 根据当前模型选择总标题。
        title = 'Logistic regression' if model_name == 'logistic' else 'Random forest'
        # 标题包含模型、分区和样本量。
        fig.suptitle(f'{title} | {split} | n={len(truth)}')
        # 保留逻辑回归原文件名，森林矩阵使用单独文件名。
        filename = f'{split}_confusion_matrix.png' if model_name == 'logistic' else f'{split}_random_forest_confusion_matrix.png'
        # 统一以300dpi保存并关闭画布。
        save_figure(fig, output_dir / filename, split)
    # 调用者用返回值汇总成绩，不必重新计算。
    return metrics


def save_comparison(output_dir, split, results):
    # 按结果字典的顺序绘制模型。
    names = list(results)
    # 每个模型都比较这三项相同指标。
    keys = ['accuracy', 'balanced_accuracy', 'macro_f1']
    # 建立一张分组柱状图。
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    # 三个横轴位置分别对应三项指标。
    x = np.arange(3)
    # 模型越多，单根柱子越窄，使每组总宽保持0.8。
    width = 0.8 / len(names)
    # 为三种方法准备固定颜色。
    colors = ['#8b9c96', '#237359', '#497eaf']
    # 每个模型画一组柱子。
    for index, name in enumerate(names):
        # 把0到1的分数转成百分数用于显示。
        values = [results[name][key] * 100 for key in keys]
        # 按模型编号向左右偏移，使整组柱子居中。
        bars = ax.bar(x + (index - (len(names) - 1) / 2) * width, values,
                      # 图例显示模型名，颜色按模型顺序分配。
                      width=width, label=name, color=colors[index % len(colors)])
        # 柱顶显示一位小数，方便直接读分数。
        ax.bar_label(bars, fmt='%.1f', padding=3)
    # 横轴使用三项指标的名称。
    ax.set(xticks=x, xticklabels=['Accuracy', 'Balanced accuracy', 'Macro-F1'],
           # 所有分数共用0到100的纵轴，避免视觉比例误导。
           ylim=(0, 100), ylabel='Score (%)', title=f'Classification methods on the same features | {split}')
    # 图例说明每种颜色对应哪个模型。
    ax.legend()
    # 淡网格方便比较柱高。
    ax.grid(axis='y', alpha=0.2)
    # 网格放在柱子后面。
    ax.set_axisbelow(True)
    # 按分区保存比较图。
    save_figure(fig, Path(output_dir) / f'{split}_model_comparison.png', split)


def save_validation_errors(output_dir, images, truth, prediction, model_name='logistic'):
    """仅展示验证集按索引排序的前12个错误，不根据test挑例或改模型。"""
    # 按原编号选前12个错误，不按错误严重程度挑选。
    indices = np.flatnonzero(truth != prediction)[:12]
    # 三行四列最多放12张图。
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), layout='constrained')
    # 子图展平后与错误编号逐个对应。
    for ax, index in zip(axes.flat, indices):
        # 展示误分类图像的原始RGB像素。
        ax.imshow(images[index])
        # 同时标出编号、真实类和预测类，方便人工检查。
        ax.set_title(f'val index={index}\ntrue={CLASS_SHORT[int(truth[index])]} | pred={CLASS_SHORT[int(prediction[index])]}', fontsize=9)
    # 包括没有填图的位置，所有子图都隐藏坐标轴。
    for ax in axes.flat:
        # 错误样本图不需要像素坐标刻度。
        ax.axis('off')
    # 没有错误时也保留一张有说明的图。
    if len(indices) == 0:
        # 把没有验证错误的提示放在第一个格子。
        axes[0, 0].text(0.5, 0.5, 'No validation errors', ha='center')
    # 标题说明模型名称和选图范围。
    fig.suptitle(f'{model_name}: first validation errors (up to 12, not representative)')
    # 各模型使用不同文件名，保留原逻辑回归输出名。
    filename = 'validation_error_examples.png' if model_name == 'logistic' else f'validation_{model_name}_error_examples.png'
    # 这里的样本全部来自验证集。
    save_figure(fig, Path(output_dir) / filename, 'val')


def save_forest_importance(output_dir, model, feature_names):
    """不纯度重要性是模型描述，不代表因果或无偏的特征贡献。"""
    # 输出目录统一转Path。
    output_dir = Path(output_dir)
    # 每一项重要性与对应的特征名称放在同一行。
    table = pd.DataFrame({'feature': feature_names, 'impurity_importance': model.feature_importances_})
    # 从大到小排序，相同重要性保留原始特征顺序。
    table = table.sort_values('impurity_importance', ascending=False, kind='stable')
    # CSV保留全部24项，图中只显示前15项。
    table.to_csv(output_dir / 'random_forest_feature_importance.csv', index=False, encoding='utf-8-sig')
    # 取前15项后倒序，让最大的一项出现在横向图顶部。
    top = table.head(15).iloc[::-1]
    # 建立适合较长特征名的横向柱状图。
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    # 横轴是不纯度重要性，纵轴是特征名称。
    ax.barh(top['feature'], top['impurity_importance'], color='#497eaf')
    # 标题提示该数值不表示因果关系。
    ax.set(xlabel='Mean decrease in impurity', title='Random forest: top 15 impurity importances (not causal)')
    # 重要性来自训练好的森林，分区脚注标为train。
    save_figure(fig, output_dir / 'random_forest_feature_importance.png', 'train')
