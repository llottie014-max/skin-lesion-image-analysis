from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from baseline import select_random_forest
from baseline_outputs import save_comparison


class RandomForestTests(unittest.TestCase):
    def test_seeded_forest_is_reproducible_and_selected_by_validation_accuracy(self):
        rng = np.random.default_rng(13)
        train = rng.normal(size=(60, 4))
        labels = (train[:, 0] > 0).astype(int)
        val = rng.normal(size=(20, 4))
        truth = (val[:, 0] > 0).astype(int)
        candidates = ({'max_depth': 2, 'min_samples_leaf': 1, 'class_weight': None},
                      {'max_depth': None, 'min_samples_leaf': 1, 'class_weight': None})
        first, selected, scores = select_random_forest(train, labels, val, truth,
                                                      candidates=candidates, n_estimators=16)
        second, repeated, _ = select_random_forest(train, labels, val, truth,
                                                   candidates=candidates, n_estimators=16)
        np.testing.assert_array_equal(first.predict(val), second.predict(val))
        self.assertEqual(selected, repeated)
        self.assertEqual(selected['random_state'], 42)
        self.assertEqual(selected['n_estimators'], 16)
        self.assertAlmostEqual(float(np.mean(first.predict(val) == truth)), max(row['accuracy'] for row in scores))
        self.assertEqual(first.n_features_in_, 4)

    def test_comparison_chart_includes_all_three_models(self):
        results = {name: {'accuracy': 0.7, 'balanced_accuracy': 0.4, 'macro_f1': 0.3}
                   for name in ['majority', 'logistic', 'random_forest']}
        with tempfile.TemporaryDirectory() as folder, patch('baseline_outputs.save_figure') as save:
            save_comparison(Path(folder), 'val', results)
            figure = save.call_args.args[0]
            try:
                labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
                self.assertEqual(labels, list(results))
                self.assertEqual(len(figure.axes[0].containers), 3)
            finally:
                plt.close(figure)


if __name__ == '__main__':
    unittest.main()
