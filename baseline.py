"""可解释分类基线：仅在train拟合，按val选择参数。"""
# 把训练时的收敛警告转成可处理的异常。
import warnings

# 用数组运算检查标签和计算特征。
import numpy as np
# 逻辑回归达到迭代上限时会发出这个警告。
from sklearn.exceptions import ConvergenceWarning
# 随机森林由多棵决策树投票分类。
from sklearn.ensemble import RandomForestClassifier
# 保留逻辑回归，方便比较换模型前后的成绩。
from sklearn.linear_model import LogisticRegression
# 同时看总体正确率、各类召回和各类F1，避免只看多数类。
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
# 把标准化和分类器放进同一个训练流程。
from sklearn.pipeline import Pipeline
# 用训练集的均值和标准差缩放特征。
from sklearn.preprocessing import StandardScaler


# 七类的编号固定为0到6，缺少的类也要计入macro-F1。
LABELS = list(range(7))
# C越小，逻辑回归的正则化约束越强。
C_GRID = (0.1, 1.0, 10.0)
# 候选顺序固定，完全平分时保留排在前面的组合。
RF_GRID = tuple(
    # 每个字典保存一组树深、叶节点样本数和类别权重。
    {'max_depth': depth, 'min_samples_leaf': leaf, 'class_weight': weight}
    # 两种深度×两种叶节点大小×两种权重，共8组。
    for depth in (12, None) for leaf in (3, 1) for weight in (None, 'balanced_subsample')
)


def evaluate_predictions(y_true, y_pred):
    """macro-F1固定七类、零除为0；balanced accuracy平均实际出现类的召回。"""
    # 统一转为数组，下面才能检查形状和数据类型。
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # 一张图像对应一个真实标签和一个预测标签。
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or len(y_true) == 0:
        # 空数组或长度不一致时，成绩没有可比性。
        raise ValueError('真实标签与预测必须是长度相同的非空一维数组。')
    # 真实标签、预测标签都检查，防止无效编号混进评估。
    for values in (y_true, y_pred):
        # 整数类型和0到6的取值范围必须同时满足。
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values > 6)):
            # 标签出错就停止，不能带着错误标签继续算分。
            raise ValueError('标签必须为0～6整数。')
    # 返回普通字典，后面可直接写入JSON或表格。
    return {
        # accuracy是预测正确的图像数除以总图像数。
        'accuracy': float(accuracy_score(y_true, y_pred)),
        # balanced accuracy对实际出现的每一类召回率取平均。
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        # macro-F1给七类相同权重，无法计算的单类F1记为0。
        'macro_f1': float(f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)),
    }


def select_model(x_train, y_train, x_val, y_val, candidates=C_GRID):
    """返回已在train拟合的最佳Pipeline、C、所有候选的val成绩；不接收test。"""
    # 候选C去重后升序排列，使平分时的选择可复现。
    candidates = sorted(set(float(value) for value in candidates))
    # 排除空列表、无穷大、NaN和非正数。
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        # C必须是有限正数，否则正则化参数无效。
        raise ValueError('候选C必须为有限的正数。')
    # 初始分数设为-1，保证第一个有效候选能被记录。
    best_model, chosen, best_score = None, None, -1.0
    # 保留所有候选的验证成绩，方便回头检查取舍。
    scores = []
    # 依次训练每个C，不使用测试集挑参数。
    for value in candidates:
        # Pipeline让缩放器和分类器按相同顺序执行。
        model = Pipeline([
            # 标准化只在下面的fit中学习训练集统计量。
            ('scale', StandardScaler()),
            # balanced按训练集类别频数调整权重，照顾少数类。
            ('classifier', LogisticRegression(C=value, class_weight='balanced',
                                             # 用lbfgs求解，最多迭代3000次，容差设为1e-4。
                                             solver='lbfgs', max_iter=3000, tol=1e-4)),
        ])
        # 临时修改警告规则，退出这个代码块后恢复。
        with warnings.catch_warnings():
            # 未收敛时直接报错，避免把不完整训练当成成功。
            warnings.simplefilter('error', ConvergenceWarning)
            # 捕获训练中可能出现的收敛问题。
            try:
                # 缩放器和逻辑回归都只接收训练集。
                model.fit(x_train, y_train)
            # 只处理未收敛警告，其他错误按原样抛出。
            except ConvergenceWarning as error:
                # 错误信息带上C，便于找到出问题的候选。
                raise RuntimeError(f'C={value}未收敛，实验停止；请检查数据与配置。') from error
        # 验证集只用于预测和评分，不参与fit。
        metrics = evaluate_predictions(y_val, model.predict(x_val))
        # 把参数和三项成绩放在同一行记录里。
        scores.append({'C': value, **metrics})
        # 严格比较未四舍五入的分数；升序C使精确平局优先较小C。
        if metrics['macro_f1'] > best_score:
            # 更新当前最高的验证集macro-F1。
            best_score = metrics['macro_f1']
            # 保留已经训练好的模型，不合并train和val重训。
            best_model, chosen = model, value
    # 返回最佳模型、对应C和全部候选成绩。
    return best_model, chosen, scores


def select_random_forest(x_train, y_train, x_val, y_val, candidates=RF_GRID,
                         n_estimators=300, verbose=False):
    """仅train拟合，按val accuracy选RF；平局比较macro-F1、balanced accuracy。"""
    # 这次以准确率为首要目标，初始时还没有最佳候选。
    best_model, selected, best_rank = None, None, None
    # 记录每组参数的成绩，后面导出CSV。
    scores = []
    # enumerate同时给出候选编号和参数字典。
    for index, candidate in enumerate(candidates):
        # 限制可调参数，避免候选悄悄覆盖随机种子等固定设置。
        if set(candidate) != {'max_depth', 'min_samples_leaf', 'class_weight'}:
            # 遇到不在计划中的参数就停止。
            raise ValueError('RF候选仅允许max_depth、min_samples_leaf、class_weight。')
        # sqrt表示每次节点分裂只随机考察部分特征。
        parameters = {'n_estimators': n_estimators, 'max_features': 'sqrt',
                      # 固定种子42，用2个工作线程，bootstrap表示有放回抽样。
                      'random_state': 42, 'n_jobs': 2, 'bootstrap': True, **candidate}
        # 每组候选只在训练集上训练一次森林。
        model = RandomForestClassifier(**parameters).fit(x_train, y_train)
        # 用当前森林预测验证集，再计算三项指标。
        metrics = evaluate_predictions(y_val, model.predict(x_val))
        # 连同编号和固定参数一起保存，便于复现实验。
        scores.append({'candidate_index': index, **parameters, **metrics})
        # 元组按顺序比较：先准确率，再macro-F1，再平衡准确率。
        rank = (metrics['accuracy'], metrics['macro_f1'], metrics['balanced_accuracy'])
        # 严格大于才更新，完全相同时保留较早的候选。
        if best_rank is None or rank > best_rank:
            # 同时更新最佳模型、参数和比较用的分数。
            best_model, selected, best_rank = model, parameters, rank
        # 命令行训练时显示进度，单元测试中可以关闭。
        if verbose:
            # 打印树深，便于对应候选表。
            print(f'  RF候选{index + 1}: depth={candidate["max_depth"]}, '
                  # 接着显示叶节点最小样本数和类别权重。
                  f'leaf={candidate["min_samples_leaf"]}, weight={candidate["class_weight"]}, '
                  # 这里的四舍五入只用于显示，不影响上面的选参。
                  f'val accuracy={metrics["accuracy"]:.4f}, macro-F1={metrics["macro_f1"]:.4f}', flush=True)
    # 防止空候选列表返回一个不能预测的None。
    if best_model is None:
        # 没有任何候选就明确报错。
        raise ValueError('RF候选列表不能为空。')
    # 调用者可以继续评估模型，也可以保存选参记录。
    return best_model, selected, scores
