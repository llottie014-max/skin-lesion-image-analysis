"""命令行入口：.venv/bin/python main.py --offline"""
import argparse
import sys

from workflow import PROJECT_DIR, download_dataset, run_analysis


def main():
    parser = argparse.ArgumentParser(description='DermaMNIST 皮肤病变图像分析与可视化', allow_abbrev=False)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train', help='分析的数据分区，默认 train')
    parser.add_argument('--offline', action='store_true', help='只使用已经缓存并通过校验的本地数据')
    parser.add_argument('--image-index', type=int, default=0, help='处理对比与直方图使用的图像编号，从 0 开始')
    args = parser.parse_args()
    print('[1/5] 准备并校验 DermaMNIST 64×64 数据……', flush=True)
    data_path = PROJECT_DIR / 'data' / 'dermamnist_64.npz'
    try:
        download_dataset(data_path, offline=args.offline)
        run_analysis(data_path, PROJECT_DIR / 'outputs' / args.split, args.split, args.image_index)
    except (OSError, ValueError) as error:
        print(f'运行未完成：{error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
