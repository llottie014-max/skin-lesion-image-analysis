import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflow import check_md5, class_statistics, download_dataset, load_dataset, to_gray


class WorkflowTests(unittest.TestCase):
    def test_offline_missing_cache_fails_without_creating_data(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'missing.npz'
            with self.assertRaises(FileNotFoundError):
                download_dataset(file, offline=True)
            self.assertFalse(file.exists())

    def test_cached_corruption_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'corrupt.npz'
            file.write_bytes(b'not a dataset')
            with self.assertRaises(ValueError):
                download_dataset(file, offline=True)

    def test_statistics_keeps_missing_classes_and_correct_percentages(self):
        table = class_statistics(np.array([0, 0, 4]))
        self.assertEqual(table['count'].tolist(), [2, 0, 0, 0, 1, 0, 0])
        self.assertAlmostEqual(table['percent'].sum(), 100.0)

    def test_gray_uses_rgb_luminance(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(to_gray(rgb), [[76, 150, 29]])

    def test_md5_detects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'file.npz'
            file.write_bytes(b'abc')
            check_md5(file, '900150983cd24fb0d6963f7d28e17f72')
            file.write_bytes(b'broken')
            with self.assertRaises(ValueError):
                check_md5(file, '900150983cd24fb0d6963f7d28e17f72')

    def test_load_rejects_invalid_labels_and_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'fixture.npz'
            images = np.zeros((2, 64, 64, 3), dtype=np.uint8)
            for labels in (np.array([[7], [0]]), np.array([[0]]), np.array([[0.5], [1]])):
                np.savez(file, train_images=images, train_labels=labels)
                with self.assertRaises(ValueError):
                    load_dataset(file, 'train')

    def test_load_flattens_labels_and_preserves_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'fixture.npz'
            images = np.full((2, 64, 64, 3), 23, dtype=np.uint8)
            np.savez(file, train_images=images, train_labels=np.array([[0], [4]]))
            loaded, labels = load_dataset(file, 'train')
            self.assertEqual(loaded.shape, (2, 64, 64, 3))
            self.assertEqual(int(loaded[0, 0, 0, 0]), 23)
            self.assertEqual(labels.tolist(), [0, 4])

    def test_missing_keys_and_empty_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'fixture.npz'
            np.savez(file, train_images=np.zeros((0, 64, 64, 3), dtype=np.uint8), train_labels=np.array([]))
            with self.assertRaises(ValueError):
                load_dataset(file, 'train')
            with self.assertRaises(ValueError):
                load_dataset(file, 'test')


if __name__ == '__main__':
    unittest.main()
