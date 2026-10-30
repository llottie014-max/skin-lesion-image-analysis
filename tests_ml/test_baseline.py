# 使用Python标准库的断言和测试发现机制。
import unittest
# 测试文件路径使用Path拼接。
from pathlib import Path
# 把测试文件放进自动清理的临时目录。
import tempfile

# 用小数组构造可手算的图像、标签和特征。
import numpy as np
# 读回导出的CSV，检查真实文件内容。
import pandas as pd

# 检查指标计算和逻辑回归选参。
from baseline import evaluate_predictions, select_model
# 读回实际输出文件，检查分类报告格式。
from baseline_outputs import save_evaluation


class BaselineTests(unittest.TestCase):
    # 检查accuracy行的总样本数，防止把准确率误填到所有列。
    def test_classification_csv_does_not_broadcast_accuracy_to_support(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            save_evaluation(folder, 'val', 'majority', np.arange(7), np.full(7, 5))
            report = pd.read_csv(Path(folder) / 'val_majority_classification.csv', index_col=0)
            self.assertEqual(report.loc['accuracy', 'support'], 7)
            self.assertTrue(pd.isna(report.loc['accuracy', 'precision']))
            self.assertTrue(pd.isna(report.loc['accuracy', 'recall']))

    # 七类各一张却始终猜同一类，三项指标应与手算一致。
    def test_majority_guess_does_not_hide_minority_failure(self):
        metrics = evaluate_predictions(np.arange(7), np.full(7, 5))
        self.assertAlmostEqual(metrics['accuracy'], 1/7)
        self.assertAlmostEqual(metrics['balanced_accuracy'], 1/7)
        self.assertAlmostEqual(metrics['macro_f1'], 1/28)

    # 验证值故意远离训练值，检查标准化均值有没有泄漏。
    def test_scaler_is_fitted_on_training_only(self):
        train = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.]])
        labels = np.array([0, 0, 1, 1])
        model, chosen, scores = select_model(train, labels, np.array([[100., 0.], [100., 2.]]),
                                              np.array([0, 1]), candidates=(1.0,))
        np.testing.assert_allclose(model.named_steps['scale'].mean_, [1., 1.])
        self.assertEqual(chosen, 1.0)
        self.assertEqual(len(scores), 1)

    # 候选分数完全相同时，逻辑回归应选较小的C。
    def test_equal_validation_scores_choose_smallest_c(self):
        train = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.]])
        model, chosen, scores = select_model(train, np.array([0, 0, 1, 1]),
                                              np.array([[0., 1.], [2., 1.]]), np.array([0, 1]))
        self.assertEqual(chosen, 0.1)
        self.assertEqual(len(scores), 3)
        self.assertEqual(model.predict([[0., 1.], [2., 1.]]).tolist(), [0, 1])

    # 长度不符、编号越界和空标签都不能计算成绩。
    def test_metric_rejects_mismatched_or_out_of_range_labels(self):
        for truth, prediction in (([0, 1], [0]), ([0, 7], [0, 1]), ([], [])):
            with self.assertRaises(ValueError):
                evaluate_predictions(np.array(truth), np.array(prediction))


# 直接运行此文件时，也能单独执行这些测试。
if __name__ == '__main__':
    # 交给unittest收集测试并输出结果。
    unittest.main()
