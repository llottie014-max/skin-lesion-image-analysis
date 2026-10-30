# 用于判断当前系统是否支持被测试的软链接结构。
import os
# 测试文件路径使用Path拼接。
from pathlib import Path
# 启动真实命令行入口，检查退出状态。
import subprocess
# 子进程使用与当前测试相同的Python解释器。
import sys
# 把测试文件放进自动清理的临时目录。
import tempfile
# 使用Python标准库的断言和测试发现机制。
import unittest

# 用小数组构造可手算的图像、标签和特征。
import numpy as np

# 直接测试只读环境检查，不执行依赖安装。
from bootstrap import inspect_environment
# 调用完整图像分析流程，检查参数校验时机。
from workflow import run_analysis


class DemoTests(unittest.TestCase):
    # 参数写成--spl时应先报错，不能已经开始读取数据。
    def test_cli_rejects_abbreviated_split_before_running(self):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'main.py'),
                                 '--spl', 'val', '--image-index', '-1', '--offline'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)

    @unittest.skipIf(os.name == 'nt', '此测试核查 macOS/POSIX 解释器软链接')
    # 只放一个普通Python软链接，不能算作项目虚拟环境。
    def test_plain_interpreter_symlink_is_not_treated_as_project_venv(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            # 当前测试的文件都写进这个临时根目录。
            root = Path(folder)
            (root / '.venv/bin').mkdir(parents=True)
            (root / '.venv/bin/python').symlink_to(Path(sys.executable).resolve())
            (root / 'requirements-lock.txt').write_text('', encoding='utf-8')
            result = inspect_environment(root)
            self.assertFalse(result['ready'])
            self.assertTrue(any('虚拟环境' in problem for problem in result['problems']))

    # 检查模式只报告环境缺失，不能顺手创建目录。
    def test_missing_environment_is_reported_without_creating_it(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            result = inspect_environment(Path(folder))
            self.assertFalse(result['ready'])
            self.assertFalse((Path(folder) / '.venv').exists())
            self.assertTrue(result['problems'])

    # 示例编号越界时，应在创建输出目录之前停止。
    def test_invalid_example_index_does_not_create_outputs(self):
        # 离开这个代码块后，测试目录和文件自动清理。
        with tempfile.TemporaryDirectory() as folder:
            # 当前测试的文件都写进这个临时根目录。
            root = Path(folder)
            source = root / 'fixture.npz'
            np.savez(source, train_images=np.zeros((2, 64, 64, 3), dtype=np.uint8),
                     train_labels=np.array([[0], [4]]))
            for index in (-1, 2):
                with self.assertRaisesRegex(ValueError, 'image-index'):
                    run_analysis(source, root / 'outputs', image_index=index)
            self.assertFalse((root / 'outputs').exists())


# 直接运行此文件时，也能单独执行这些测试。
if __name__ == '__main__':
    # 交给unittest收集测试并输出结果。
    unittest.main()
