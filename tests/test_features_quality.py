import unittest

import numpy as np

from features import FEATURE_NAMES, extract_features
from quality import audit_splits


class FeaturesQualityTests(unittest.TestCase):
    def test_red_image_has_hand_checked_color_gray_and_histogram(self):
        images = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        images[:, :, :, 0] = 255
        result = extract_features(images)
        self.assertEqual(result.shape, (1, 24))
        self.assertEqual(len(FEATURE_NAMES), 24)
        np.testing.assert_allclose(result[0, :8], [1, 0, 0, 0, 0, 0, 76/255, 0])
        self.assertAlmostEqual(result[0, 8:].sum(), 1)
        self.assertEqual(result[0, 8 + 4], 1)

    def test_features_reject_float_or_non_rgb_data(self):
        for images in (np.zeros((1, 2, 2, 3)), np.zeros((1, 2, 2), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                extract_features(images)

    def test_hash_audit_distinguishes_duplicate_groups_from_extra_images(self):
        a = np.zeros((2, 2, 3), dtype=np.uint8)
        b = np.ones((2, 2, 3), dtype=np.uint8)
        report = audit_splits({
            'train': (np.array([a, a, b]), np.array([0, 0, 1])),
            'val': (np.array([a]), np.array([4])),
            'test': (np.array([b]), np.array([1])),
        })
        self.assertEqual(report['within_split']['train']['duplicate_groups'], 1)
        self.assertEqual(report['within_split']['train']['extra_images'], 1)
        self.assertEqual(report['cross_split_shared_groups'], 2)
        self.assertEqual(report['conflicting_label_groups'], 1)
        self.assertEqual(report['split_sizes'], {'train': 3, 'val': 1, 'test': 1})


if __name__ == '__main__':
    unittest.main()
