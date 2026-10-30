"""按预先固定的官方分区协议训练、评估和记录教学基线。"""
# 给每次实验记录带时区的UTC时间。
from datetime import datetime, timezone
# 读取实际安装的库版本，方便复现。
import importlib.metadata
# 将完整参数和成绩写入JSON。
import json
# 组合输出文件路径。
from pathlib import Path
# 记录运行实验时的Python版本。
import platform

# 检查类别覆盖情况并统计每类样本数。
import numpy as np
# 导出候选参数和模型系数表。
import pandas as pd
# 多数类模型不看图像内容，用作最低限度的比较。
from sklearn.dummy import DummyClassifier

# 两个分类器各自按预先确定的验证指标选参。
from baseline import C_GRID, select_model, select_random_forest
# 输出表格、对比图、错误样本和森林重要性。
from baseline_outputs import save_comparison, save_evaluation, save_validation_errors, save_forest_importance
# 特征名称与提取顺序由同一个模块定义。
from features import FEATURE_NAMES, extract_features
# 先报告精确像素重复，不据此偷偷删除样本。
from quality import audit_splits
# 复用官方数据常量、文件校验和读取函数。
from workflow import DATA_MD5, DATA_URL, SPLIT_COUNTS, check_md5, configure_font, load_dataset


def run_experiment(data_path, output_dir, evaluate_test=False, include_random_forest=False):
    """保留官方标签与划分；审计只报告问题，不悄悄清理数据。"""
    # 完整文件校验通过后才读取数据。
    check_md5(data_path, DATA_MD5)
    # 按官方名称分别读取三个分区，保持原划分。
    splits = {name: load_dataset(data_path, name) for name in SPLIT_COUNTS}
    # 逐个核对分区数量和类别覆盖情况。
    for name, (images, labels) in splits.items():
        # 本实验要求每个分区数量正确，并覆盖0到6七类。
        if len(images) != SPLIT_COUNTS[name] or np.unique(labels).tolist() != list(range(7)):
            # 不符合预期时停止，防止使用残缺数据。
            raise ValueError(f'{name}应有{SPLIT_COUNTS[name]}张图像，并覆盖全部七类。')
    # 输出位置统一转为Path。
    output_dir = Path(output_dir)
    # 父目录不存在时一并创建。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 设置绘图字体，避免中文无法显示。
    configure_font()
    # 先提示审计步骤，再运行计算。
    print('[1/4] 检查原始RGB像素的精确重复与标签冲突……', flush=True)
    # 检查精确重复与同像素不同标签的情况。
    quality = audit_splits(splits)
    # 保存完整审计记录，便于定位重复图像。
    (output_dir / 'data_quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # 打印共享组数，同时提醒病灶级重复尚未解决。
    print(f'  跨分区共享像素组：{quality["cross_split_shared_groups"]}；不代表已完成病灶级去重。', flush=True)
    # 说明从此处开始提取模型输入。
    print('[2/4] 提取24维特征；仅在train拟合缩放器与模型……', flush=True)
    # 训练集转成(N,24)的特征矩阵。
    x_train = extract_features(splits['train'][0])
    # 训练集标签与特征逐行对应。
    y_train = splits['train'][1]
    # 验证集使用相同定义提取特征，不重新学习规则。
    x_val = extract_features(splits['val'][0])
    # 验证标签只用于选参与评估。
    y_val = splits['val'][1]
    # 返回已经训练好的逻辑回归、最佳C和所有候选成绩。
    model, chosen_c, candidates = select_model(x_train, y_train, x_val, y_val)
    # 只从训练标签学习出现最多的类别。
    dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
    # 默认不开森林，先为可选结果留空。
    forest, forest_parameters, forest_candidates = None, None, []
    # 只有打开随机森林开关才额外训练。
    if include_random_forest:
        # 提示候选数量和选参目标。
        print('  随机森林：8个预先固定的候选，仅按val accuracy选参……', flush=True)
        # 森林的拟合和验证选参在专门函数里完成。
        forest, forest_parameters, forest_candidates = select_random_forest(
            # 只传train和val，选参函数拿不到test。
            x_train, y_train, x_val, y_val, verbose=True)
    # 显示逻辑回归选中的C，并开始保存验证结果。
    print(f'[3/4] 按val macro-F1选择 C={chosen_c}，保存验证集评估……', flush=True)
    # 最佳逻辑回归对验证集作一次预测。
    val_predictions = model.predict(x_val)
    # 按分区、模型两层字典保存指标。
    metrics = {'val': {
        # 保存多数类基线的逐图预测和指标。
        'majority': save_evaluation(output_dir, 'val', 'majority', y_val, dummy.predict(x_val)),
        # 保存逻辑回归验证结果，便于与多数类基线比较。
        'logistic': save_evaluation(output_dir, 'val', 'logistic', y_val, val_predictions),
    }}
    # 关闭RF时直接跳过这一组输出。
    if forest is not None:
        # 最佳森林预测验证集，不在验证集上继续训练。
        forest_predictions = forest.predict(x_val)
        # RF沿用同一个评估函数，指标定义一致。
        metrics['val']['random_forest'] = save_evaluation(output_dir, 'val', 'random_forest', y_val, forest_predictions)
        # 展示按编号排序的森林验证错误。
        save_validation_errors(output_dir, splits['val'][0], y_val, forest_predictions, 'random_forest')
        # 导出森林的不纯度重要性，只作模型描述。
        save_forest_importance(output_dir, forest, FEATURE_NAMES)
        # 8个候选全部保留，包含未被选中的组合。
        pd.DataFrame(forest_candidates).to_csv(output_dir / 'random_forest_candidates.csv', index=False, encoding='utf-8-sig')
    # 将当前启用的所有模型画在同一张验证对比图中。
    save_comparison(output_dir, 'val', metrics['val'])
    # 逻辑回归错误样本单独保留，便于对照。
    save_validation_errors(output_dir, splits['val'][0], y_val, val_predictions)
    # 只有显式打开test评估开关才进入此分支。
    if evaluate_test:
        # 模型与C已经固定；test不参与特征定义、拟合、参数选择或重训。
        # 提示测试集只用于评估已经固定的模型。
        print('  显式开启test：只评估已经选定并固定的train拟合模型。', flush=True)
        # 这时才计算测试集特征，仍使用相同24维定义。
        x_test = extract_features(splits['test'][0])
        # 取出测试集原有标签用于评分。
        y_test = splits['test'][1]
        # 测试成绩单独放在test键下。
        metrics['test'] = {
            # 多数类预测也沿用训练集学到的类别。
            'majority': save_evaluation(output_dir, 'test', 'majority', y_test, dummy.predict(x_test)),
            # 逻辑回归只调用predict，不再调用fit。
            'logistic': save_evaluation(output_dir, 'test', 'logistic', y_test, model.predict(x_test)),
        }
        # 只有训练过森林才输出RF测试结果。
        if forest is not None:
            # 同一棵已选定森林预测test，不根据test更换参数。
            metrics['test']['random_forest'] = save_evaluation(output_dir, 'test', 'random_forest', y_test, forest.predict(x_test))
        # 用同一套绘图规则比较测试集成绩。
        save_comparison(output_dir, 'test', metrics['test'])
    # 另存逻辑回归的验证候选，保留原基线文件名。
    pd.DataFrame(candidates).to_csv(output_dir / 'validation_candidates.csv', index=False, encoding='utf-8-sig')
    # 系数的列顺序与24维特征名称相同。
    pd.DataFrame(model.named_steps['classifier'].coef_, columns=FEATURE_NAMES,
                 # 行号使用模型的实际类别编号，防止系数对应错类。
                 index=model.classes_).to_csv(output_dir / 'model_coefficients.csv', index_label='class_label', encoding='utf-8-sig')
    # 下面的摘要同时记录输入、训练协议、成绩和环境。
    summary = {
        # 使用UTC时间，跨机器查看时不受本地时区影响。
        'created_utc': datetime.now(timezone.utc).isoformat(),
        # 保留原官方划分及其病灶重叠局限说明。
        'protocol': 'Original official DermaMNIST split; educational benchmark with known lesion-overlap limitations; not clinical validation',
        # 数据来源和MD5可用于核对输入文件。
        'source_url': DATA_URL, 'data_md5': DATA_MD5,
        # 记录三个分区实际读取到的图像数。
        'split_sizes': {name: len(images) for name, (images, _) in splits.items()},
        # 记录各分区七类样本数，方便查看类别不平衡。
        'class_support': {name: np.bincount(labels, minlength=7).tolist() for name, (_, labels) in splits.items()},
        # 特征名称和维数与模型输入保持一致。
        'feature_names': FEATURE_NAMES, 'feature_count': len(FEATURE_NAMES),
        # 写明缩放、标准差和直方图边界的定义。
        'feature_definition': 'RGB mean/std per channel and Pillow L gray mean/std divided by 255; ddof=0; 16 gray histogram proportions with fixed edges 0,16,...,256',
        # 缩放器、分类器只拟合train，选参只使用val。
        'fit_split': 'train', 'scaler_fit_split': 'train', 'selection_split': 'val',
        # 逻辑回归按未四舍五入的macro-F1选C，完全平分时选较小C。
        'selection_rule': 'maximum unrounded validation macro-F1 over 7 fixed labels; exact ties use smallest C; no train+val refit',
        # 保留候选列表、最终选择及每个候选的成绩。
        'candidate_C': list(C_GRID), 'selected_C': chosen_c, 'validation_candidates': candidates,
        # 记录逻辑回归固定的求解与类别权重设置。
        'classifier': {'solver': 'lbfgs', 'class_weight': 'balanced', 'max_iter': 3000, 'tol': 1e-4},
        # 明确记录test是否开启，避免误把val成绩当test成绩。
        'test_evaluated': evaluate_test, 'metrics': metrics,
        # 保存训练集学到的标准化均值。
        'scaler_mean': model.named_steps['scale'].mean_.tolist(),
        # 保存标准化尺度，用于解释特征如何被缩放。
        'scaler_scale': model.named_steps['scale'].scale_.tolist(),
        # 模型类别顺序决定系数和截距的对应关系。
        'coefficient_class_order': model.classes_.tolist(),
        # 保存每类逻辑回归的截距。
        'intercept': model.named_steps['classifier'].intercept_.tolist(),
        # 记录Python版本。
        'python_version': platform.python_version(),
        # 记录实际运行环境的第三方库版本。
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'Pillow', 'scikit-learn', 'scipy', 'matplotlib']},
        # 摘要只摘取审计关键计数，完整明细在data_quality.json中。
        'quality_summary': {key: quality[key] for key in ['cross_split_shared_groups', 'conflicting_label_groups', 'within_split']},
    }
    # RF未开启时，摘要中不添加不存在的模型信息。
    if forest is not None:
        # 随机森林的规则单独记录，避免与逻辑回归选参目标混淆。
        summary['random_forest'] = {
            # 记录最终选中的全部森林参数。
            'parameters': forest_parameters,
            # 保留所有森林验证候选。
            'validation_candidates': forest_candidates,
            # 写明准确率优先、平局规则以及仅train拟合。
            'selection_rule': 'maximum validation accuracy; ties: macro-F1, balanced accuracy, then predefined candidate order; train-only fitting without scaling',
            # 重要性可能有偏，不能解释为特征的因果作用。
            'importance_note': 'Impurity importance can be biased and is not a causal or clinical explanation.',
            # test此前已经查看过，本轮只能算同分区探索比较。
            'evaluation_note': 'Original official test was examined in earlier baseline experiments; this is a same-split exploratory comparison, not a new independent holdout.',
        }
    # 将结构化摘要写入文件，保留中文说明。
    (output_dir / 'experiment.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # 准备一份能直接阅读的Markdown结果表。
    rows = ['# 教学分类基线结果', '',
            # 报告开头说明分区重叠和像素审计的局限。
            '本结果使用原官方分区，存在已报告的同病灶跨集合局限；像素级重复检查不等于病灶级去重。', '',
            # 显示特征维度、选中的C和测试集评估开关。
            f'24维特征；仅train拟合；val macro-F1选C={chosen_c}；test评估开启：{evaluate_test}。', '',
            # 表格依次列出三项评价指标。
            '| 分区 | 方法 | Accuracy | Balanced accuracy | Macro-F1 |',
            # Markdown表格的列对齐格式。
            '|---|---|---:|---:|---:|']
    # 依次写入每个已评估分区。
    for split, results in metrics.items():
        # 同一分区下逐个写入模型成绩。
        for name, scores in results.items():
            # 展示保留4位小数，模型选择仍使用原始分数。
            rows.append(f'| {split} | {name} | {scores["accuracy"]:.4f} | {scores["balanced_accuracy"]:.4f} | {scores["macro_f1"]:.4f} |')
    # 表后给出候选参数和完整配置文件的位置。
    rows += ['', '参数选择与验证结果见validation_candidates.csv；完整配置和版本见experiment.json。',
             # 错误图只展示前12个错误，不代表全部错误的分布。
             '错误示例仅取验证集按编号排序的前12个错误，不代表总体错误分布。',
             # 提醒后续实验不能反复根据test改参数。
             '不能根据test分数反复改特征或调参；增加方法时应预先确定协议。',
             # 保留数据来源和许可，方便作业引用。
             '数据来源：MedMNIST / HAM10000，CC BY-NC 4.0，完整引用见项目DATA_SOURCE.md。', '']
    # 启用森林时，在报告末尾补充RF实验说明。
    if forest is not None:
        # 写明森林的准确率选参和平局规则。
        rows += ['随机森林按val accuracy选参；平局依次比较macro-F1、balanced accuracy和固定候选顺序。',
                 # 指出候选表位置，并解释重要性的含义边界。
                 '候选见random_forest_candidates.csv；不纯度特征重要性不是因果解释。',
                 # 保留测试集已被查看过的事实。
                 'test已在前轮基线实验查看，本次是相同原分区的探索对比，不是全新独立验证。', '']
    # 将文本行用换行符连接后保存。
    (output_dir / 'results.md').write_text('\n'.join(rows), encoding='utf-8')
    # 完成时显示输出目录。
    print(f'[4/4] 实验完成：{output_dir}', flush=True)
    # 返回摘要，供单元测试和其他脚本直接检查。
    return summary
