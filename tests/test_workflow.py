# 把测试文件放进自动清理的临时目录。
import tempfile
# 使用Python标准库的断言和测试发现机制。
import unittest
# 测试文件路径使用Path拼接。
from pathlib import Path

# 用小数组构造可手算的图像、标签和特征。
import numpy as np

# 分别测试校验、下载、读取、统计和灰度转换。
from workflow import check_md5, class_statistics, download_dataset, load_dataset, to_gray


class WorkflowTests(unittest.TestCase):
    # 离线且没有缓存时，应报错并保持数据路径不存在。
    def test_offline_missing_cache_fails_without_creating_data(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'missing.npz'
            with self.assertRaises(FileNotFoundError):
                download_dataset(file, offline=True)
            self.assertFalse(file.exists())

    # 同名文件即使存在，也必须通过校验才能使用。
    def test_cached_corruption_is_not_silently_accepted(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'corrupt.npz'
            file.write_bytes(b'not a dataset')
            with self.assertRaises(ValueError):
                download_dataset(file, offline=True)

    # 缺少某些类时，统计表仍应有七行，比例总和为100%。
    def test_statistics_keeps_missing_classes_and_correct_percentages(self):
        table = class_statistics(np.array([0, 0, 4]))
        self.assertEqual(table['count'].tolist(), [2, 0, 0, 0, 1, 0, 0])
        self.assertAlmostEqual(table['percent'].sum(), 100.0)

    # 红绿蓝纯色的灰度值不同，用已知结果检查亮度权重。
    def test_gray_uses_rgb_luminance(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(to_gray(rgb), [[76, 150, 29]])

    # 同一文件先写正确内容再改坏，确认校验能发现变化。
    def test_md5_detects_corruption(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'file.npz'
            file.write_bytes(b'abc')
            check_md5(file, '900150983cd24fb0d6963f7d28e17f72')
            file.write_bytes(b'broken')
            with self.assertRaises(ValueError):
                check_md5(file, '900150983cd24fb0d6963f7d28e17f72')

    # 分别测试越界标签、数量不符和浮点标签。
    def test_load_rejects_invalid_labels_and_mismatched_lengths(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'fixture.npz'
            images = np.zeros((2, 64, 64, 3), dtype=np.uint8)
            for labels in (np.array([[7], [0]]), np.array([[0]]), np.array([[0.5], [1]])):
                np.savez(file, train_images=images, train_labels=labels)
                with self.assertRaises(ValueError):
                    load_dataset(file, 'train')

    # 读取后标签可以展平，但原始图像像素不能被改动。
    def test_load_flattens_labels_and_preserves_pixels(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'fixture.npz'
            images = np.full((2, 64, 64, 3), 23, dtype=np.uint8)
            np.savez(file, train_images=images, train_labels=np.array([[0], [4]]))
            loaded, labels = load_dataset(file, 'train')
            self.assertEqual(loaded.shape, (2, 64, 64, 3))
            self.assertEqual(int(loaded[0, 0, 0, 0]), 23)
            self.assertEqual(labels.tolist(), [0, 4])

    # NPZ缺少所选分区或图像为空时，都应停止读取。
    def test_missing_keys_and_empty_images_are_rejected(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'fixture.npz'
            np.savez(file, train_images=np.zeros((0, 64, 64, 3), dtype=np.uint8), train_labels=np.array([]))
            with self.assertRaises(ValueError):
                load_dataset(file, 'train')
            with self.assertRaises(ValueError):
                load_dataset(file, 'test')


# 直接运行此文件时，也能单独执行这些测试。
if __name__ == '__main__':
    # 交给unittest收集测试并输出结果。
    unittest.main()
