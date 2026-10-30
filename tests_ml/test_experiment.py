# 为临时数据文件计算对应的MD5校验值。
import hashlib
# 读取实验输出的JSON报告。
import json
# 测试文件路径使用Path拼接。
from pathlib import Path
# 把测试文件放进自动清理的临时目录。
import tempfile
# 使用Python标准库的断言和测试发现机制。
import unittest
# 只替换测试边界所需的常量或文件保存操作。
from unittest.mock import patch

# 用小数组构造可手算的图像、标签和特征。
import numpy as np
# 读回导出的CSV，检查真实文件内容。
import pandas as pd

# 测试中临时替换此模块引用的数据大小和校验值。
import experiment


class ExperimentTests(unittest.TestCase):
    # 开启RF后应输出三个模型，默认仍不输出测试集成绩。
    def test_random_forest_mode_exports_three_models_without_test_evaluation(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            # 当前测试的文件都写进这个临时根目录。
            root = Path(folder)
            # 用小型合成数据测试流程，不下载官方数据。
            source = root / 'toy.npz'
            # 七类各用一个编号，便于检查输出顺序。
            labels = np.arange(7, dtype=np.uint8)
            # 每类使用不同亮度的纯色图，七类都能被覆盖。
            images = np.array([np.full((64, 64, 3), index * 30, dtype=np.uint8) for index in labels])
            # 训练集每类放两张，共14张。
            np.savez(source, train_images=np.repeat(images, 2, axis=0),
                     # 标签按图像重复顺序排列，保持(N,1)形状。
                     train_labels=np.repeat(labels, 2).reshape(-1, 1),
                     # 验证集七张，一类一张。
                     val_images=images, val_labels=labels.reshape(-1, 1),
                     # 测试集也保留七张，仅用于检查默认不评估test。
                     test_images=images, test_labels=labels.reshape(-1, 1))
            # 只在测试范围内用小样本数量替换官方数量。
            with patch.dict(experiment.SPLIT_COUNTS, {'train': 14, 'val': 7, 'test': 7}), \
                 patch.object(experiment, 'DATA_MD5', hashlib.md5(source.read_bytes()).hexdigest()):
                # 运行真实的RF训练与输出流程，不替换分类器。
                summary = experiment.run_experiment(source, root / 'results', include_random_forest=True)
            # RF模式应同时保留多数类、逻辑回归和森林三种方法。
            self.assertEqual(list(summary['metrics']['val']), ['majority', 'logistic', 'random_forest'])
            # 默认模式必须明确记录未评估测试集。
            self.assertFalse(summary['test_evaluated'])
            # 除了摘要，也检查磁盘上没有test开头的输出文件。
            self.assertFalse(any((root / 'results').glob('test_*')))
            # 8组固定候选都应有成绩记录。
            self.assertEqual(len(summary['random_forest']['validation_candidates']), 8)
            # 从磁盘读回重要性表，检查实际导出结果。
            importance = pd.read_csv(root / 'results/random_forest_feature_importance.csv')
            # 重要性表应保留全部24个特征。
            self.assertEqual(len(importance), 24)
            # 这个可分裂的合成数据中，归一化重要性之和应为1。
            self.assertAlmostEqual(importance['impurity_importance'].sum(), 1)
            # 读回森林逐图预测表。
            predictions = pd.read_csv(root / 'results/val_random_forest_predictions.csv')
            # 输出标签顺序应与输入七张图的顺序相同。
            self.assertEqual(predictions['true_label'].tolist(), list(range(7)))
            # RF混淆矩阵图应实际写入文件。
            self.assertTrue((root / 'results/val_random_forest_confusion_matrix.png').is_file())

    # 默认入口只评估val，同时保留选参、预测和审计记录。
    def test_default_run_outputs_validation_only_and_records_selection(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            # 当前测试的文件都写进这个临时根目录。
            root = Path(folder)
            # 用小型合成数据测试流程，不下载官方数据。
            source = root / 'toy.npz'
            # 七类各用一个编号，便于检查输出顺序。
            labels = np.arange(7, dtype=np.uint8)
            # 每类使用不同亮度的纯色图，七类都能被覆盖。
            images = np.array([np.full((64, 64, 3), index * 30, dtype=np.uint8) for index in labels])
            # 训练集每类放两张，共14张。
            np.savez(source, train_images=np.repeat(images, 2, axis=0),
                     # 标签按图像重复顺序排列，保持(N,1)形状。
                     train_labels=np.repeat(labels, 2).reshape(-1, 1),
                     # 验证集七张，一类一张。
                     val_images=images, val_labels=labels.reshape(-1, 1),
                     # 测试集也保留七张，仅用于检查默认不评估test。
                     test_images=images, test_labels=labels.reshape(-1, 1))
            # 只在测试范围内用小样本数量替换官方数量。
            with patch.dict(experiment.SPLIT_COUNTS, {'train': 14, 'val': 7, 'test': 7}), \
                 patch.object(experiment, 'DATA_MD5', hashlib.md5(source.read_bytes()).hexdigest()):
                # 不加开关，检查原逻辑回归默认流程。
                summary = experiment.run_experiment(source, root / 'results')
            # 默认模式必须明确记录未评估测试集。
            self.assertFalse(summary['test_evaluated'])
            # 默认成绩字典只能包含验证集。
            self.assertEqual(list(summary['metrics']), ['val'])
            # 参数选择必须记录为使用val。
            self.assertEqual(summary['selection_split'], 'val')
            # 三个逻辑回归C值都应被记录。
            self.assertEqual(len(summary['validation_candidates']), 3)
            # 除了摘要，也检查磁盘上没有test开头的输出文件。
            self.assertFalse(any((root / 'results').glob('test_*')))
            # 读回逻辑回归逐图预测表。
            predictions = pd.read_csv(root / 'results/val_logistic_predictions.csv')
            # 输出标签顺序应与输入七张图的顺序相同。
            self.assertEqual(predictions['true_label'].tolist(), list(range(7)))
            # 读取实际写出的审计报告。
            audit = json.loads((root / 'results/data_quality.json').read_text())
            # 合成数据刻意跨分区重复，应该报告七个共享像素组。
            self.assertEqual(audit['cross_split_shared_groups'], 7)


# 直接运行此文件时，也能单独执行这些测试。
if __name__ == '__main__':
    # 交给unittest收集测试并输出结果。
    unittest.main()
