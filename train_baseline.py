"""可选机器学习入口：python train_baseline.py --offline。"""
import argparse
import sys

from workflow import PROJECT_DIR, download_dataset


def main():
    parser = argparse.ArgumentParser(description='皮肤病变图像教学分类基线', allow_abbrev=False)
    parser.add_argument('--offline', action='store_true', help='仅使用本地已校验数据')
    parser.add_argument('--evaluate-test', action='store_true', help='选定模型后显式评估原官方test；不用于调参')
    parser.add_argument('--random-forest', action='store_true', help='增加随机森林，按val准确率选参并与两种原基线比较')
    args = parser.parse_args()
    try:
        from experiment import run_experiment
    except ModuleNotFoundError as error:
        if error.name not in {'sklearn', 'scipy', 'joblib', 'threadpoolctl', 'narwhals', 'cloudpickle'}:
            raise
        print(f'缺少机器学习依赖 {error.name}，请运行：.venv/bin/python -m pip install -r requirements-ml-lock.txt', file=sys.stderr)
        return 1
    folder = 'test_evaluation' if args.evaluate_test else 'validation'
    experiment_folder = 'random_forest' if args.random_forest else 'baseline'
    try:
        data = download_dataset(PROJECT_DIR / 'data/dermamnist_64.npz', offline=args.offline)
        run_experiment(data, PROJECT_DIR / 'outputs' / experiment_folder / folder,
                       args.evaluate_test, include_random_forest=args.random_forest)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'实验未完成：{error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
