# 测试文件路径使用Path拼接。
from pathlib import Path
# 把测试文件放进自动清理的临时目录。
import tempfile
# 使用Python标准库的断言和测试发现机制。
import unittest

# 根据当前系统构造解释器路径，再测试环境检查。
from bootstrap import environment_python, inspect_environment


class BootstrapMLTests(unittest.TestCase):
    # 机器学习检查不能误用普通锁文件，也不能自动安装。
    def test_ml_profile_requires_its_own_lockfile_without_installing(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            # 当前测试的文件都写进这个临时根目录。
            root = Path(folder)
            python = environment_python(root)
            python.parent.mkdir(parents=True)
            python.write_bytes(b'')
            (root / 'requirements-lock.txt').write_text('', encoding='utf-8')
            status = inspect_environment(root, ml=True)
            self.assertFalse(status['ready'])
            self.assertTrue(any('requirements-ml-lock.txt' in problem for problem in status['problems']))
            self.assertFalse((root / 'requirements-ml-lock.txt').exists())


# 直接运行此文件时，也能单独执行这些测试。
if __name__ == '__main__':
    # 交给unittest收集测试并输出结果。
    unittest.main()
