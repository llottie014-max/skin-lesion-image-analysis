"""按预先固定的官方分区协议训练、评估和记录教学基线。"""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from baseline import C_GRID, select_model, select_random_forest
from baseline_outputs import save_comparison, save_evaluation, save_validation_errors, save_forest_importance
from features import FEATURE_NAMES, extract_features
from quality import audit_splits
from workflow import DATA_MD5, DATA_URL, SPLIT_COUNTS, check_md5, configure_font, load_dataset


def run_experiment(data_path, output_dir, evaluate_test=False, include_random_forest=False):
    """保留官方标签与划分；审计只报告问题，不悄悄清理数据。"""
    check_md5(data_path, DATA_MD5)
    splits = {name: load_dataset(data_path, name) for name in SPLIT_COUNTS}
    for name, (images, labels) in splits.items():
        if len(images) != SPLIT_COUNTS[name] or np.unique(labels).tolist() != list(range(7)):
            raise ValueError(f'{name}应有{SPLIT_COUNTS[name]}张图像，并覆盖全部七类。')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_font()
    print('[1/4] 检查原始RGB像素的精确重复与标签冲突……', flush=True)
    quality = audit_splits(splits)
    (output_dir / 'data_quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'  跨分区共享像素组：{quality["cross_split_shared_groups"]}；不代表已完成病灶级去重。', flush=True)
    print('[2/4] 提取24维特征；仅在train拟合缩放器与模型……', flush=True)
    x_train = extract_features(splits['train'][0])
    y_train = splits['train'][1]
    x_val = extract_features(splits['val'][0])
    y_val = splits['val'][1]
    model, chosen_c, candidates = select_model(x_train, y_train, x_val, y_val)
    dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
    forest, forest_parameters, forest_candidates = None, None, []
    if include_random_forest:
        print('  随机森林：8个预先固定的候选，仅按val accuracy选参……', flush=True)
        forest, forest_parameters, forest_candidates = select_random_forest(
            x_train, y_train, x_val, y_val, verbose=True)
    print(f'[3/4] 按val macro-F1选择 C={chosen_c}，保存验证集评估……', flush=True)
    val_predictions = model.predict(x_val)
    metrics = {'val': {
        'majority': save_evaluation(output_dir, 'val', 'majority', y_val, dummy.predict(x_val)),
        'logistic': save_evaluation(output_dir, 'val', 'logistic', y_val, val_predictions),
    }}
    if forest is not None:
        forest_predictions = forest.predict(x_val)
        metrics['val']['random_forest'] = save_evaluation(output_dir, 'val', 'random_forest', y_val, forest_predictions)
        save_validation_errors(output_dir, splits['val'][0], y_val, forest_predictions, 'random_forest')
        save_forest_importance(output_dir, forest, FEATURE_NAMES)
        pd.DataFrame(forest_candidates).to_csv(output_dir / 'random_forest_candidates.csv', index=False, encoding='utf-8-sig')
    save_comparison(output_dir, 'val', metrics['val'])
    save_validation_errors(output_dir, splits['val'][0], y_val, val_predictions)
    if evaluate_test:
        # 模型与C已经固定；test不参与特征定义、拟合、参数选择或重训。
        print('  显式开启test：只评估已经选定并固定的train拟合模型。', flush=True)
        x_test = extract_features(splits['test'][0])
        y_test = splits['test'][1]
        metrics['test'] = {
            'majority': save_evaluation(output_dir, 'test', 'majority', y_test, dummy.predict(x_test)),
            'logistic': save_evaluation(output_dir, 'test', 'logistic', y_test, model.predict(x_test)),
        }
        if forest is not None:
            metrics['test']['random_forest'] = save_evaluation(output_dir, 'test', 'random_forest', y_test, forest.predict(x_test))
        save_comparison(output_dir, 'test', metrics['test'])
    pd.DataFrame(candidates).to_csv(output_dir / 'validation_candidates.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(model.named_steps['classifier'].coef_, columns=FEATURE_NAMES,
                 index=model.classes_).to_csv(output_dir / 'model_coefficients.csv', index_label='class_label', encoding='utf-8-sig')
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': 'Original official DermaMNIST split; educational benchmark with known lesion-overlap limitations; not clinical validation',
        'source_url': DATA_URL, 'data_md5': DATA_MD5,
        'split_sizes': {name: len(images) for name, (images, _) in splits.items()},
        'class_support': {name: np.bincount(labels, minlength=7).tolist() for name, (_, labels) in splits.items()},
        'feature_names': FEATURE_NAMES, 'feature_count': len(FEATURE_NAMES),
        'feature_definition': 'RGB mean/std per channel and Pillow L gray mean/std divided by 255; ddof=0; 16 gray histogram proportions with fixed edges 0,16,...,256',
        'fit_split': 'train', 'scaler_fit_split': 'train', 'selection_split': 'val',
        'selection_rule': 'maximum unrounded validation macro-F1 over 7 fixed labels; exact ties use smallest C; no train+val refit',
        'candidate_C': list(C_GRID), 'selected_C': chosen_c, 'validation_candidates': candidates,
        'classifier': {'solver': 'lbfgs', 'class_weight': 'balanced', 'max_iter': 3000, 'tol': 1e-4},
        'test_evaluated': evaluate_test, 'metrics': metrics,
        'scaler_mean': model.named_steps['scale'].mean_.tolist(),
        'scaler_scale': model.named_steps['scale'].scale_.tolist(),
        'coefficient_class_order': model.classes_.tolist(),
        'intercept': model.named_steps['classifier'].intercept_.tolist(),
        'python_version': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'Pillow', 'scikit-learn', 'scipy', 'matplotlib']},
        'quality_summary': {key: quality[key] for key in ['cross_split_shared_groups', 'conflicting_label_groups', 'within_split']},
    }
    if forest is not None:
        summary['random_forest'] = {
            'parameters': forest_parameters,
            'validation_candidates': forest_candidates,
            'selection_rule': 'maximum validation accuracy; ties: macro-F1, balanced accuracy, then predefined candidate order; train-only fitting without scaling',
            'importance_note': 'Impurity importance can be biased and is not a causal or clinical explanation.',
            'evaluation_note': 'Original official test was examined in earlier baseline experiments; this is a same-split exploratory comparison, not a new independent holdout.',
        }
    (output_dir / 'experiment.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    rows = ['# 教学分类基线结果', '',
            '本结果使用原官方分区，存在已报告的同病灶跨集合局限；像素级重复检查不等于病灶级去重。', '',
            f'24维特征；仅train拟合；val macro-F1选C={chosen_c}；test评估开启：{evaluate_test}。', '',
            '| 分区 | 方法 | Accuracy | Balanced accuracy | Macro-F1 |',
            '|---|---|---:|---:|---:|']
    for split, results in metrics.items():
        for name, scores in results.items():
            rows.append(f'| {split} | {name} | {scores["accuracy"]:.4f} | {scores["balanced_accuracy"]:.4f} | {scores["macro_f1"]:.4f} |')
    rows += ['', '参数选择与验证结果见validation_candidates.csv；完整配置和版本见experiment.json。',
             '错误示例仅取验证集按编号排序的前12个错误，不代表总体错误分布。',
             '不能根据test分数反复改特征或调参；增加方法时应预先确定协议。',
             '数据来源：MedMNIST / HAM10000，CC BY-NC 4.0，完整引用见项目DATA_SOURCE.md。', '']
    if forest is not None:
        rows += ['随机森林按val accuracy选参；平局依次比较macro-F1、balanced accuracy和固定候选顺序。',
                 '候选见random_forest_candidates.csv；不纯度特征重要性不是因果解释。',
                 'test已在前轮基线实验查看，本次是相同原分区的探索对比，不是全新独立验证。', '']
    (output_dir / 'results.md').write_text('\n'.join(rows), encoding='utf-8')
    print(f'[4/4] 实验完成：{output_dir}', flush=True)
    return summary
