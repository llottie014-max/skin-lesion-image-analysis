import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import experiment


class ExperimentTests(unittest.TestCase):
    def test_default_run_outputs_validation_only_and_records_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'toy.npz'
            labels = np.arange(7, dtype=np.uint8)
            images = np.array([np.full((64, 64, 3), index * 30, dtype=np.uint8) for index in labels])
            np.savez(source, train_images=np.repeat(images, 2, axis=0),
                     train_labels=np.repeat(labels, 2).reshape(-1, 1),
                     val_images=images, val_labels=labels.reshape(-1, 1),
                     test_images=images, test_labels=labels.reshape(-1, 1))
            with patch.dict(experiment.SPLIT_COUNTS, {'train': 14, 'val': 7, 'test': 7}), \
                 patch.object(experiment, 'DATA_MD5', hashlib.md5(source.read_bytes()).hexdigest()):
                summary = experiment.run_experiment(source, root / 'results')
            self.assertFalse(summary['test_evaluated'])
            self.assertEqual(list(summary['metrics']), ['val'])
            self.assertEqual(summary['selection_split'], 'val')
            self.assertEqual(len(summary['validation_candidates']), 3)
            self.assertFalse(any((root / 'results').glob('test_*')))
            predictions = pd.read_csv(root / 'results/val_logistic_predictions.csv')
            self.assertEqual(predictions['true_label'].tolist(), list(range(7)))
            audit = json.loads((root / 'results/data_quality.json').read_text())
            self.assertEqual(audit['cross_split_shared_groups'], 7)


if __name__ == '__main__':
    unittest.main()
