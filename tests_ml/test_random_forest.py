# 测试文件路径使用Path拼接。
from pathlib import Path
# 把测试文件放进自动清理的临时目录。
import tempfile
# 使用Python标准库的断言和测试发现机制。
import unittest
# 只替换测试边界所需的常量或文件保存操作。
from unittest.mock import patch

# 检查画布后关闭它，避免测试残留图形。
import matplotlib.pyplot as plt
# 用小数组构造可手算的图像、标签和特征。
import numpy as np

# 使用真实的小型森林验证选参和复现行为。
from baseline import select_random_forest
# 绘制实际比较图，检查所有模型是否都出现。
from baseline_outputs import save_comparison


class RandomForestTests(unittest.TestCase):
    # 重复训练检查种子可复现，再确认所选模型的val准确率最高。
    def test_seeded_forest_is_reproducible_and_selected_by_validation_accuracy(self):
        # 固定合成数据的种子，使测试输入可重复。
        rng = np.random.default_rng(13)
        # 训练集只有60行、4维，测试运行更快。
        train = rng.normal(size=(60, 4))
        # 第一维大于0记为类1，否则为类0。
        labels = (train[:, 0] > 0).astype(int)
        # 验证特征重新生成，不直接复用训练行。
        val = rng.normal(size=(20, 4))
        # 验证标签沿用同一条已知规则。
        truth = (val[:, 0] > 0).astype(int)
        # 两个候选只改变树深，便于检查选择规则。
        candidates = ({'max_depth': 2, 'min_samples_leaf': 1, 'class_weight': None},
                      # 第二组不限制深度，其他参数保持一致。
                      {'max_depth': None, 'min_samples_leaf': 1, 'class_weight': None})
        # 第一次训练，同时保存所选参数和候选成绩。
        first, selected, scores = select_random_forest(train, labels, val, truth,
                                                      # 测试用16棵树缩短时间，实际实验默认300棵。
                                                      candidates=candidates, n_estimators=16)
        # 相同输入再训练一次，检查输出是否可复现。
        second, repeated, _ = select_random_forest(train, labels, val, truth,
                                                   # 测试用16棵树缩短时间，实际实验默认300棵。
                                                   candidates=candidates, n_estimators=16)
        # 两次预测必须逐项完全相同。
        np.testing.assert_array_equal(first.predict(val), second.predict(val))
        # 相同种子下，选中的整组参数也应一致。
        self.assertEqual(selected, repeated)
        # 确认函数使用固定的森林随机种子。
        self.assertEqual(selected['random_state'], 42)
        # 确认调用时指定的小树数确实生效。
        self.assertEqual(selected['n_estimators'], 16)
        # 重新计算所选模型准确率，应等于候选中的最高值。
        self.assertAlmostEqual(float(np.mean(first.predict(val) == truth)), max(row['accuracy'] for row in scores))
        # 模型实际接收四个特征，确认输入维数没有变化。
        self.assertEqual(first.n_features_in_, 4)

    # 检查图例与柱组，防止新加的森林被绘图循环漏掉。
    def test_comparison_chart_includes_all_three_models(self):
        # 用相同分数构造三种方法，专门检查图形结构。
        results = {name: {'accuracy': 0.7, 'balanced_accuracy': 0.4, 'macro_f1': 0.3}
                   # 新加入的森林也必须进入绘图循环。
                   for name in ['majority', 'logistic', 'random_forest']}
        # 只替换最终存盘函数，图表仍然实际绘制。
        with tempfile.TemporaryDirectory() as folder, patch('baseline_outputs.save_figure') as save:
            # 绘制包含三种方法的验证集对比图。
            save_comparison(Path(folder), 'val', results)
            # 从保存函数收到的参数中取出真实画布。
            figure = save.call_args.args[0]
            try:
                # 读取实际图例文字，不能只检查输入字典。
                labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
                # 图例必须与三个模型的名称和顺序一致。
                self.assertEqual(labels, list(results))
                # 每个模型一个柱组，应有三组。
                self.assertEqual(len(figure.axes[0].containers), 3)
            finally:
                # 这里替换了保存函数，需要手动关闭画布。
                plt.close(figure)


# 直接运行此文件时，也能单独执行这些测试。
if __name__ == '__main__':
    # 交给unittest收集测试并输出结果。
    unittest.main()
