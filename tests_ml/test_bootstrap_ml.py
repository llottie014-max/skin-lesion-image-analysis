from pathlib import Path
import tempfile
import unittest

from bootstrap import environment_python, inspect_environment


class BootstrapMLTests(unittest.TestCase):
    def test_ml_profile_requires_its_own_lockfile_without_installing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = environment_python(root)
            python.parent.mkdir(parents=True)
            python.write_bytes(b'')
            (root / 'requirements-lock.txt').write_text('', encoding='utf-8')
            status = inspect_environment(root, ml=True)
            self.assertFalse(status['ready'])
            self.assertTrue(any('requirements-ml-lock.txt' in problem for problem in status['problems']))
            self.assertFalse((root / 'requirements-ml-lock.txt').exists())


if __name__ == '__main__':
    unittest.main()
