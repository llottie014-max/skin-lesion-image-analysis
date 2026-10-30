"""只用标准库配置项目环境：python3 bootstrap.py；--check 只检查。"""
# 解析--check、--json和--ml选项。
import argparse
# 在父进程与检查子进程之间传递结构化结果。
import json
# 区分Windows与macOS/Linux的环境目录结构。
import os
# 组合项目和解释器路径。
from pathlib import Path
# 在PATH里查找已安装的Python。
import shutil
# 创建虚拟环境、安装依赖和运行独立检查。
import subprocess
# 获取当前解释器路径，输出错误与退出码。
import sys


# 以脚本位置确定项目根目录。
PROJECT_DIR = Path(__file__).resolve().parent


def environment_python(root):
    # Windows解释器在Scripts下，其他系统在bin下。
    return Path(root) / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def inspect_environment(root, ml=False):
    """用子进程检查解释器、锁定版本和实际导入，不修改文件。"""
    # 把传入的根目录统一转为Path。
    root = Path(root)
    # 找到本项目预期使用的解释器位置。
    python = environment_python(root)
    # 先假定环境未就绪，逐项检查后再更新。
    result = {'ready': False, 'interpreter': str(python), 'problems': []}
    # 文件缺失或软链接失效时无法启动检查。
    if not python.is_file():
        # 将缺少环境的原因加入问题列表。
        result['problems'].append('项目虚拟环境缺失或解释器链接失效。')
        # 只检查不创建，调用者可据此决定是否配置。
        return result
    # 机器学习模式使用自己的完整锁文件。
    lock = root / ('requirements-ml-lock.txt' if ml else 'requirements-lock.txt')
    # 锁文件缺失时不能确认依赖版本。
    if not lock.is_file():
        # 报错中保留具体文件名。
        result['problems'].append(f'缺少 {lock.name}，无法核对依赖。')
        # 缺少锁文件时提前返回，避免误报环境就绪。
        return result
    # 保存依赖名称和要求的精确版本。
    requirements = {}
    # 逐行读取锁文件。
    for line in lock.read_text(encoding='utf-8').splitlines():
        # 跳过空行和注释行。
        if line.strip() and not line.startswith('#'):
            # 锁文件每项以包名==版本书写，只分割一次。
            name, version = line.split('==', 1)
            # 形成检查子进程需要的版本字典。
            requirements[name] = version
    # 这段字符串交给项目解释器执行，检查的是目标环境而非当前进程。
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
    # 解释器无法启动、超时或返回无效JSON都应记作检查失败。
    try:
        # 机器学习模式额外测试sklearn和scipy是否能真正导入。
        imports = ['numpy', 'pandas', 'matplotlib', 'PIL'] + (['sklearn', 'scipy'] if ml else [])
        # 用目标解释器运行上面的检查脚本，并传入依赖版本。
        completed = subprocess.run([str(python), '-c', probe, json.dumps(requirements),
                                    # 同时传入预期环境路径与需要测试导入的模块。
                                    str(root / '.venv'), json.dumps(imports)],
                                   # 收集标准输出和错误输出，最多等待90秒。
                                   text=True, capture_output=True, timeout=90)
        # 非零退出码说明检查子进程没有正常完成。
        if completed.returncode:
            # 优先保留解释器自己的错误信息。
            result['problems'].append(completed.stderr.strip() or '解释器检查失败。')
        # 退出码正常时才解析检查报告。
        else:
            # 将包版本、解释器身份和问题列表写回结果。
            result.update(json.loads(completed.stdout))
    # 将启动失败、超时和JSON格式错误统一记录。
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        # 保留具体异常文本，方便定位环境问题。
        result['problems'].append(str(error))
    # 只有问题列表为空，才认为环境就绪。
    result['ready'] = not result['problems']
    # 返回检查报告，不在此处修改环境。
    return result


def find_python312():
    """寻找已安装的 3.12；不安装或更改系统解释器。"""
    # 先找当前解释器，再找PATH里的python3.12。
    candidates = [sys.executable, shutil.which('python3.12'),
                  # 兼容本机工具附带的Python运行时位置。
                  str(Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'),
                  # 再尝试macOS常见的Python官方安装位置。
                  '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3']
    # 去掉空路径和重复项，避免反复检查同一个解释器。
    for candidate in dict.fromkeys(value for value in candidates if value):
        # 每个候选独立检查，失败后继续尝试下一个。
        try:
            # 直接运行候选解释器，确认主次版本确实为3.12。
            result = subprocess.run([candidate, '-c', 'import sys; print(sys.version_info[:2] == (3, 12))'],
                                    # 只读取文本输出，最多等待15秒。
                                    capture_output=True, text=True, timeout=15)
            # 启动成功且输出True，才接受这个候选。
            if result.returncode == 0 and result.stdout.strip() == 'True':
                # 返回可用于创建虚拟环境的解释器路径。
                return candidate
        # 路径无效或进程卡住时，尝试其他候选。
        except (OSError, subprocess.TimeoutExpired):
            # 跳过当前不可用的解释器。
            continue
    # 全部候选都不可用时，给出安装提示。
    raise RuntimeError('未找到可用 Python 3.12。请先安装 3.12，再运行配置；不会改动系统 Python。')


def ensure_environment(root, ml=False):
    """健康环境直接复用；缺环境则创建，缺依赖则按锁文件安装。"""
    # 统一项目根目录类型。
    root = Path(root)
    # 先检查已有环境，避免不必要的重复安装。
    status = inspect_environment(root, ml=ml)
    # 版本和导入都通过时直接复用。
    if status['ready']:
        # 返回现成的检查报告。
        return status
    # 找到将要使用或创建的项目解释器位置。
    python = environment_python(root)
    # 解释器不存在时，才考虑创建新环境。
    if not python.is_file():
        # 如果.venv目录已经存在，先保护原内容。
        if (root / '.venv').exists():
            # 不自动删除可能包含用户文件的旧目录。
            raise RuntimeError('.venv 已存在但不可用，请先将该目录改名保存，再重新配置。')
        # 创建前显示当前操作。
        print('创建项目 Python 3.12 环境……', flush=True)
        # 使用确认过的3.12解释器执行标准库venv。
        subprocess.run([find_python312(), '-m', 'venv', str(root / '.venv')], check=True)
    # 已有解释器也必须属于本项目虚拟环境。
    elif not status.get('is_project_venv'):
        # 拒绝用普通解释器软链接冒充独立环境。
        raise RuntimeError('现有解释器不属于项目独立虚拟环境，请先改名保存 .venv，再配置。')
    # 已存在的环境必须是3.12，不能只核对路径。
    elif status.get('python_version', '').split('.')[:2] != ['3', '12']:
        # 版本不对时停止，保留旧环境供使用者处理。
        raise RuntimeError('现有虚拟环境不是可用的 Python 3.12，请先改名保存 .venv，再配置。')
    # 安装前提示需要联网。
    print('按锁定版本安装项目依赖（需要网络）……', flush=True)
    # 根据是否启用机器学习选择对应的锁文件。
    lock = root / ('requirements-ml-lock.txt' if ml else 'requirements-lock.txt')
    # 通过项目解释器调用pip，避免安装进其他Python。
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(lock)],
                   # 安装失败直接抛异常，工作目录固定为项目目录。
                   check=True, cwd=root)
    # 安装完成后重新检查版本与实际导入。
    status = inspect_environment(root, ml=ml)
    # pip退出成功也不等于环境一定可用，还要看复查结果。
    if not status['ready']:
        # 把所有未通过的检查项合并到错误信息中。
        raise RuntimeError('安装后检查未通过：' + '；'.join(status['problems']))
    # 返回确认可用的环境状态。
    return status


def main():
    # 不接受参数缩写，避免把输错的参数误识别。
    parser = argparse.ArgumentParser(description='自动配置项目独立 Python 环境', allow_abbrev=False)
    # 只检查时不创建环境、不安装依赖。
    parser.add_argument('--check', action='store_true', help='只检查，不安装或修改环境')
    # JSON格式便于其他脚本读取检查结果。
    parser.add_argument('--json', action='store_true', help='以 JSON 输出检查结果（与 --check 配合）')
    # 启用机器学习依赖的配置或检查。
    parser.add_argument('--ml', action='store_true', help='同时配置/检查可选的基础分类实验依赖')
    # 解析调用者输入的选项。
    args = parser.parse_args()
    # JSON报告仅用于检查模式。
    if args.json and not args.check:
        # 参数组合不合要求时由argparse退出。
        parser.error('--json 需要同时指定 --check')
    # 集中处理配置期间可以说明的常见错误。
    try:
        # --check选择检查函数，否则检查并补齐环境。
        status = inspect_environment(PROJECT_DIR, ml=args.ml) if args.check else ensure_environment(PROJECT_DIR, ml=args.ml)
    # 捕获路径、安装子进程、版本和锁文件等问题。
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        # 把具体配置失败原因写到stderr。
        print('环境配置失败：' + str(error), file=sys.stderr)
        # 非零退出码表示配置失败。
        return 1
    # 检查脚本要求JSON时，只输出结构化报告。
    if args.json:
        # 保留中文并缩进，便于人和程序共同阅读。
        print(json.dumps(status, ensure_ascii=False, indent=2))
    # 普通输出下，环境通过时显示解释器版本。
    elif status['ready']:
        # 确认版本和导入检查均已通过。
        print('环境就绪：Python ' + status['python_version'] + '；锁定依赖均可导入。')
    # 环境未就绪时显示所有已发现的问题。
    else:
        # 用中文分号连接多个问题，便于逐项排查。
        print('环境未就绪：' + '；'.join(status['problems']))
    # 就绪返回0，未就绪返回1。
    return 0 if status['ready'] else 1


# 被其他文件导入时，不执行配置操作。
if __name__ == '__main__':
    # 把配置结果作为进程退出状态。
    sys.exit(main())
