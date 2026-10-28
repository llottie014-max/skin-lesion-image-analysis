"""保存分类评估表与科研风格静态图，不生成网页。"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from baseline import LABELS, evaluate_predictions
from workflow import CLASS_NAMES, CLASS_SHORT, configure_font, save_figure


def save_evaluation(output_dir, split, model_name, truth, prediction):
    output_dir = Path(output_dir)
    metrics = evaluate_predictions(truth, prediction)
    pd.DataFrame({
        'split': split, 'image_index': np.arange(len(truth)),
        'true_label': truth, 'predicted_label': prediction,
        'correct': truth == prediction,
    }).to_csv(output_dir / f'{split}_{model_name}_predictions.csv', index=False, encoding='utf-8-sig')
    report = classification_report(truth, prediction, labels=LABELS,
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    accuracy = report.pop('accuracy', None)
    table = pd.DataFrame.from_dict(report, orient='index')
    if accuracy is not None:
        # accuracy是标量，不能广播为precision/recall或误填到support。
        table.loc['accuracy'] = {'precision': np.nan, 'recall': np.nan,
                                 'f1-score': accuracy, 'support': len(truth)}
    table.to_csv(output_dir / f'{split}_{model_name}_classification.csv', encoding='utf-8-sig')
    matrix = confusion_matrix(truth, prediction, labels=LABELS)
    pd.DataFrame(matrix, index=CLASS_SHORT, columns=CLASS_SHORT).to_csv(
        output_dir / f'{split}_{model_name}_confusion.csv', encoding='utf-8-sig', index_label='true_class')
    if model_name == 'logistic':
        support = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
        for ax, values, title, normalized_flag in zip(
                axes, [matrix, normalized], ['Counts', 'Recall by true class'], [False, True]):
            ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized_flag else max(1, matrix.max()))
            ax.set(xticks=LABELS, yticks=LABELS, xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT,
                   xlabel='Predicted label', ylabel='True label', title=title)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            for row in LABELS:
                for column in LABELS:
                    text = f'{values[row, column]:.2f}' if normalized_flag else str(values[row, column])
                    threshold = 0.5 if normalized_flag else max(1, matrix.max()) / 2
                    ax.text(column, row, text, ha='center', va='center', fontsize=8,
                            color='white' if values[row, column] > threshold else '#172c27')
        fig.suptitle(f'Logistic regression | {split} | n={len(truth)}')
        save_figure(fig, output_dir / f'{split}_confusion_matrix.png', split)
    return metrics


def save_comparison(output_dir, split, results):
    names = list(results)
    keys = ['accuracy', 'balanced_accuracy', 'macro_f1']
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    x = np.arange(3)
    for index, (name, color) in enumerate(zip(names, ['#8b9c96', '#237359'])):
        values = [results[name][key] * 100 for key in keys]
        bars = ax.bar(x + (index - 0.5) * 0.34, values, width=0.34, label=name, color=color)
        ax.bar_label(bars, fmt='%.1f', padding=3)
    ax.set(xticks=x, xticklabels=['Accuracy', 'Balanced accuracy', 'Macro-F1'],
           ylim=(0, 100), ylabel='Score (%)', title=f'Majority guess vs learned features | {split}')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    save_figure(fig, Path(output_dir) / f'{split}_model_comparison.png', split)


def save_validation_errors(output_dir, images, truth, prediction):
    """仅展示验证集按索引排序的前12个错误，不根据test挑例或改模型。"""
    indices = np.flatnonzero(truth != prediction)[:12]
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), layout='constrained')
    for ax, index in zip(axes.flat, indices):
        ax.imshow(images[index])
        ax.set_title(f'val index={index}\ntrue={CLASS_SHORT[int(truth[index])]} | pred={CLASS_SHORT[int(prediction[index])]}', fontsize=9)
    for ax in axes.flat:
        ax.axis('off')
    if len(indices) == 0:
        axes[0, 0].text(0.5, 0.5, 'No validation errors', ha='center')
    fig.suptitle('First validation errors by index (up to 12, not representative)')
    save_figure(fig, Path(output_dir) / 'validation_error_examples.png', 'val')
