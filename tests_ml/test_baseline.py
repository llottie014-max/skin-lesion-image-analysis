import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from baseline import evaluate_predictions, select_model
from baseline_outputs import save_evaluation


class BaselineTests(unittest.TestCase):
    def test_classification_csv_does_not_broadcast_accuracy_to_support(self):
        with tempfile.TemporaryDirectory() as folder:
            save_evaluation(folder, 'val', 'majority', np.arange(7), np.full(7, 5))
            report = pd.read_csv(Path(folder) / 'val_majority_classification.csv', index_col=0)
            self.assertEqual(report.loc['accuracy', 'support'], 7)
            self.assertTrue(pd.isna(report.loc['accuracy', 'precision']))
            self.assertTrue(pd.isna(report.loc['accuracy', 'recall']))

    def test_majority_guess_does_not_hide_minority_failure(self):
        metrics = evaluate_predictions(np.arange(7), np.full(7, 5))
        self.assertAlmostEqual(metrics['accuracy'], 1/7)
        self.assertAlmostEqual(metrics['balanced_accuracy'], 1/7)
        self.assertAlmostEqual(metrics['macro_f1'], 1/28)

    def test_scaler_is_fitted_on_training_only(self):
        train = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.]])
        labels = np.array([0, 0, 1, 1])
        model, chosen, scores = select_model(train, labels, np.array([[100., 0.], [100., 2.]]),
                                              np.array([0, 1]), candidates=(1.0,))
        np.testing.assert_allclose(model.named_steps['scale'].mean_, [1., 1.])
        self.assertEqual(chosen, 1.0)
        self.assertEqual(len(scores), 1)

    def test_equal_validation_scores_choose_smallest_c(self):
        train = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.]])
        model, chosen, scores = select_model(train, np.array([0, 0, 1, 1]),
                                              np.array([[0., 1.], [2., 1.]]), np.array([0, 1]))
        self.assertEqual(chosen, 0.1)
        self.assertEqual(len(scores), 3)
        self.assertEqual(model.predict([[0., 1.], [2., 1.]]).tolist(), [0, 1])

    def test_metric_rejects_mismatched_or_out_of_range_labels(self):
        for truth, prediction in (([0, 1], [0]), ([0, 7], [0, 1]), ([], [])):
            with self.assertRaises(ValueError):
                evaluate_predictions(np.array(truth), np.array(prediction))


if __name__ == '__main__':
    unittest.main()
