"""命令行入口：.venv/bin/python main.py --offline"""
# 读取终端里输入的选项。
import argparse
# 把错误写到stderr，并把退出状态交给系统。
import sys

# 入口只负责参数和调度，图像分析放在workflow中。
from workflow import PROJECT_DIR, download_dataset, run_analysis


def main():
    # 不允许缩写参数，避免把拼错的选项误当成有效输入。
    parser = argparse.ArgumentParser(description='DermaMNIST 皮肤病变图像分析与可视化', allow_abbrev=False)
    # 默认分析训练集，也可以显式选择val或test。
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train', help='分析的数据分区，默认 train')
    # 离线模式只读取已通过校验的本地数据。
    parser.add_argument('--offline', action='store_true', help='只使用已经缓存并通过校验的本地数据')
    # 编号从0开始，指定用于处理对比的那张图像。
    parser.add_argument('--image-index', type=int, default=0, help='处理对比与直方图使用的图像编号，从 0 开始')
    # 解析选项，拼错或类型不符时由argparse报错。
    args = parser.parse_args()
    # flush让进度立即出现在终端中。
    print('[1/5] 准备并校验 DermaMNIST 64×64 数据……', flush=True)
    # 用脚本所在目录定位数据，换启动目录也能找到文件。
    data_path = PROJECT_DIR / 'data' / 'dermamnist_64.npz'
    # 集中处理下载、读取和分析中的常见错误。
    try:
        # 存在缓存就校验，缺缓存且允许联网时才下载。
        download_dataset(data_path, offline=args.offline)
        # 不同分区分别保存结果，避免互相覆盖。
        run_analysis(data_path, PROJECT_DIR / 'outputs' / args.split, args.split, args.image_index)
    # 错误信息保留具体原因，方便排查文件或参数问题。
    except (OSError, ValueError) as error:
        # 错误输出与正常进度分开。
        print(f'运行未完成：{error}', file=sys.stderr)
        # 非零退出码表示本次运行失败。
        return 1
    # 0表示分析正常结束。
    return 0


# 只有直接运行脚本时才执行入口，导入模块时不运行。
if __name__ == '__main__':
    # 将main返回的0或1作为进程退出码。
    sys.exit(main())
