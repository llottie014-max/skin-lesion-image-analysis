import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from bootstrap import inspect_environment
from workflow import run_analysis


class DemoTests(unittest.TestCase):
    def test_cli_rejects_abbreviated_split_before_running(self):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'main.py'),
                                 '--spl', 'val', '--image-index', '-1', '--offline'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)

    @unittest.skipIf(os.name == 'nt', '此测试核查 macOS/POSIX 解释器软链接')
    def test_plain_interpreter_symlink_is_not_treated_as_project_venv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '.venv/bin').mkdir(parents=True)
            (root / '.venv/bin/python').symlink_to(Path(sys.executable).resolve())
            (root / 'requirements-lock.txt').write_text('', encoding='utf-8')
            result = inspect_environment(root)
            self.assertFalse(result['ready'])
            self.assertTrue(any('虚拟环境' in problem for problem in result['problems']))

    def test_missing_environment_is_reported_without_creating_it(self):
        with tempfile.TemporaryDirectory() as folder:
            result = inspect_environment(Path(folder))
            self.assertFalse(result['ready'])
            self.assertFalse((Path(folder) / '.venv').exists())
            self.assertTrue(result['problems'])

    def test_invalid_example_index_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'fixture.npz'
            np.savez(source, train_images=np.zeros((2, 64, 64, 3), dtype=np.uint8),
                     train_labels=np.array([[0], [4]]))
            for index in (-1, 2):
                with self.assertRaisesRegex(ValueError, 'image-index'):
                    run_analysis(source, root / 'outputs', image_index=index)
            self.assertFalse((root / 'outputs').exists())


if __name__ == '__main__':
    unittest.main()
