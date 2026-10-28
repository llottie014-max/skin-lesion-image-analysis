"""只用标准库配置项目环境：python3 bootstrap.py；--check 只检查。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent


def environment_python(root):
    return Path(root) / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def inspect_environment(root, ml=False):
    """用子进程检查解释器、锁定版本和实际导入，不修改文件。"""
    root = Path(root)
    python = environment_python(root)
    result = {'ready': False, 'interpreter': str(python), 'problems': []}
    if not python.is_file():
        result['problems'].append('项目虚拟环境缺失或解释器链接失效。')
        return result
    lock = root / ('requirements-ml-lock.txt' if ml else 'requirements-lock.txt')
    if not lock.is_file():
        result['problems'].append(f'缺少 {lock.name}，无法核对依赖。')
        return result
    requirements = {}
    for line in lock.read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            name, version = line.split('==', 1)
            requirements[name] = version
    probe = '''
import importlib.metadata
import json
from pathlib import Path
import sys
required = json.loads(sys.argv[1])
problems = []
packages = {}
expected_prefix = Path(sys.argv[2]).resolve()
is_project_venv = (sys.prefix != sys.base_prefix
                   and Path(sys.prefix).resolve() == expected_prefix
                   and (expected_prefix / 'pyvenv.cfg').is_file())
if not is_project_venv:
    print(json.dumps({'python_version': sys.version.split()[0], 'is_project_venv': False,
                      'packages': {}, 'problems': ['解释器不是本项目的独立虚拟环境。']}))
    sys.exit(0)
if sys.version_info[:2] != (3, 12):
    problems.append('项目要求 Python 3.12，当前为 ' + sys.version.split()[0])
for name, expected in required.items():
    try:
        version = importlib.metadata.version(name)
        packages[name] = version
        if version != expected:
            problems.append(name + ': ' + version + '，锁定版本 ' + expected)
    except importlib.metadata.PackageNotFoundError:
        problems.append('缺少依赖：' + name)
for module in json.loads(sys.argv[3]):
    try:
        __import__(module)
    except Exception as error:
        problems.append(module + ' 导入失败：' + str(error))
print(json.dumps({'python_version': sys.version.split()[0], 'is_project_venv': True, 'packages': packages, 'problems': problems}))
'''
    try:
        imports = ['numpy', 'pandas', 'matplotlib', 'PIL'] + (['sklearn', 'scipy'] if ml else [])
        completed = subprocess.run([str(python), '-c', probe, json.dumps(requirements),
                                    str(root / '.venv'), json.dumps(imports)],
                                   text=True, capture_output=True, timeout=90)
        if completed.returncode:
            result['problems'].append(completed.stderr.strip() or '解释器检查失败。')
        else:
            result.update(json.loads(completed.stdout))
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        result['problems'].append(str(error))
    result['ready'] = not result['problems']
    return result


def find_python312():
    """寻找已安装的 3.12；不安装或更改系统解释器。"""
    candidates = [sys.executable, shutil.which('python3.12'),
                  str(Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'),
                  '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3']
    for candidate in dict.fromkeys(value for value in candidates if value):
        try:
            result = subprocess.run([candidate, '-c', 'import sys; print(sys.version_info[:2] == (3, 12))'],
                                    capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip() == 'True':
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError('未找到可用 Python 3.12。请先安装 3.12，再运行配置；不会改动系统 Python。')


def ensure_environment(root, ml=False):
    """健康环境直接复用；缺环境则创建，缺依赖则按锁文件安装。"""
    root = Path(root)
    status = inspect_environment(root, ml=ml)
    if status['ready']:
        return status
    python = environment_python(root)
    if not python.is_file():
        if (root / '.venv').exists():
            raise RuntimeError('.venv 已存在但不可用，请先将该目录改名保存，再重新配置。')
        print('创建项目 Python 3.12 环境……', flush=True)
        subprocess.run([find_python312(), '-m', 'venv', str(root / '.venv')], check=True)
    elif not status.get('is_project_venv'):
        raise RuntimeError('现有解释器不属于项目独立虚拟环境，请先改名保存 .venv，再配置。')
    elif status.get('python_version', '').split('.')[:2] != ['3', '12']:
        raise RuntimeError('现有虚拟环境不是可用的 Python 3.12，请先改名保存 .venv，再配置。')
    print('按锁定版本安装项目依赖（需要网络）……', flush=True)
    lock = root / ('requirements-ml-lock.txt' if ml else 'requirements-lock.txt')
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(lock)],
                   check=True, cwd=root)
    status = inspect_environment(root, ml=ml)
    if not status['ready']:
        raise RuntimeError('安装后检查未通过：' + '；'.join(status['problems']))
    return status


def main():
    parser = argparse.ArgumentParser(description='自动配置项目独立 Python 环境', allow_abbrev=False)
    parser.add_argument('--check', action='store_true', help='只检查，不安装或修改环境')
    parser.add_argument('--json', action='store_true', help='以 JSON 输出检查结果（与 --check 配合）')
    parser.add_argument('--ml', action='store_true', help='同时配置/检查可选的基础分类实验依赖')
    args = parser.parse_args()
    if args.json and not args.check:
        parser.error('--json 需要同时指定 --check')
    try:
        status = inspect_environment(PROJECT_DIR, ml=args.ml) if args.check else ensure_environment(PROJECT_DIR, ml=args.ml)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print('环境配置失败：' + str(error), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    elif status['ready']:
        print('环境就绪：Python ' + status['python_version'] + '；锁定依赖均可导入。')
    else:
        print('环境未就绪：' + '；'.join(status['problems']))
    return 0 if status['ready'] else 1


if __name__ == '__main__':
    sys.exit(main())
