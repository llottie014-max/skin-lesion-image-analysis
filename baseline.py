"""可解释分类基线：仅在train拟合，按val选择参数。"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = list(range(7))
C_GRID = (0.1, 1.0, 10.0)
RF_GRID = tuple(
    {'max_depth': depth, 'min_samples_leaf': leaf, 'class_weight': weight}
    for depth in (12, None) for leaf in (3, 1) for weight in (None, 'balanced_subsample')
)


def evaluate_predictions(y_true, y_pred):
    """macro-F1固定七类、零除为0；balanced accuracy平均实际出现类的召回。"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or len(y_true) == 0:
        raise ValueError('真实标签与预测必须是长度相同的非空一维数组。')
    for values in (y_true, y_pred):
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values > 6)):
            raise ValueError('标签必须为0～6整数。')
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)),
    }


def select_model(x_train, y_train, x_val, y_val, candidates=C_GRID):
    """返回已在train拟合的最佳Pipeline、C、所有候选的val成绩；不接收test。"""
    candidates = sorted(set(float(value) for value in candidates))
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError('候选C必须为有限的正数。')
    best_model, chosen, best_score = None, None, -1.0
    scores = []
    for value in candidates:
        model = Pipeline([
            ('scale', StandardScaler()),
            ('classifier', LogisticRegression(C=value, class_weight='balanced',
                                             solver='lbfgs', max_iter=3000, tol=1e-4)),
        ])
        # 不把未收敛的模型默认为成功，也不静默隐藏警告。
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            try:
                model.fit(x_train, y_train)
            except ConvergenceWarning as error:
                raise RuntimeError(f'C={value}未收敛，实验停止；请检查数据与配置。') from error
        metrics = evaluate_predictions(y_val, model.predict(x_val))
        scores.append({'C': value, **metrics})
        # 严格比较未四舍五入的分数；升序C使精确平局优先较小C。
        if metrics['macro_f1'] > best_score:
            best_score = metrics['macro_f1']
            best_model, chosen = model, value
    return best_model, chosen, scores


def select_random_forest(x_train, y_train, x_val, y_val, candidates=RF_GRID,
                         n_estimators=300, verbose=False):
    """仅train拟合，按val accuracy选RF；平局比较macro-F1、balanced accuracy。"""
    best_model, selected, best_rank = None, None, None
    scores = []
    for index, candidate in enumerate(candidates):
        if set(candidate) != {'max_depth', 'min_samples_leaf', 'class_weight'}:
            raise ValueError('RF候选仅允许max_depth、min_samples_leaf、class_weight。')
        parameters = {'n_estimators': n_estimators, 'max_features': 'sqrt',
                      'random_state': 42, 'n_jobs': 2, 'bootstrap': True, **candidate}
        model = RandomForestClassifier(**parameters).fit(x_train, y_train)
        metrics = evaluate_predictions(y_val, model.predict(x_val))
        scores.append({'candidate_index': index, **parameters, **metrics})
        rank = (metrics['accuracy'], metrics['macro_f1'], metrics['balanced_accuracy'])
        if best_rank is None or rank > best_rank:
            best_model, selected, best_rank = model, parameters, rank
        if verbose:
            print(f'  RF候选{index + 1}: depth={candidate["max_depth"]}, '
                  f'leaf={candidate["min_samples_leaf"]}, weight={candidate["class_weight"]}, '
                  f'val accuracy={metrics["accuracy"]:.4f}, macro-F1={metrics["macro_f1"]:.4f}', flush=True)
    if best_model is None:
        raise ValueError('RF候选列表不能为空。')
    return best_model, selected, scores
