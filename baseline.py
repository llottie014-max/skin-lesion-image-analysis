"""可解释分类基线：仅在train拟合，按val选择参数。"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = list(range(7))
C_GRID = (0.1, 1.0, 10.0)


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
