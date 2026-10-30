"""可选机器学习入口：python train_baseline.py --offline。"""
# 解析训练命令的选项。
import argparse
# 用于报错和设置程序退出码。
import sys

# 数据定位与下载沿用图像分析入口的实现。
from workflow import PROJECT_DIR, download_dataset


# 训练入口默认只评估验证集。
def main():
    # 参数必须写全，避免误触发其他选项。
    parser = argparse.ArgumentParser(description='皮肤病变图像教学分类基线', allow_abbrev=False)
    # 要求使用已有且通过校验的数据文件。
    parser.add_argument('--offline', action='store_true', help='仅使用本地已校验数据')
    # 只有显式开启时，才输出测试集成绩。
    parser.add_argument('--evaluate-test', action='store_true', help='选定模型后显式评估原官方test；不用于调参')
    # 打开随机森林对比，原来的逻辑回归仍会运行。
    parser.add_argument('--random-forest', action='store_true', help='增加随机森林，按val准确率选参并与两种原基线比较')
    # 把命令行选项解析成args对象。
    args = parser.parse_args()
    # 延后导入可选依赖，没装机器学习库时也能显示安装提示。
    try:
        # 实验模块串起训练、选参、评估和输出。
        from experiment import run_experiment
    # 捕获缺模块问题，区分依赖没装和代码自身导入错误。
    except ModuleNotFoundError as error:
        # 只对已知机器学习依赖给安装提示。
        if error.name not in {'sklearn', 'scipy', 'joblib', 'threadpoolctl', 'narwhals', 'cloudpickle'}:
            # 其他缺模块问题直接抛出，保留排错线索。
            raise
        # 打印项目锁文件的安装命令，确保版本一致。
        print(f'缺少机器学习依赖 {error.name}，请运行：.venv/bin/python -m pip install -r requirements-ml-lock.txt', file=sys.stderr)
        # 缺少依赖时返回失败状态。
        return 1
    # 验证集和测试集评估写进不同子目录。
    folder = 'test_evaluation' if args.evaluate_test else 'validation'
    # 随机森林结果单独保存，保留原基线输出。
    experiment_folder = 'random_forest' if args.random_forest else 'baseline'
    # 数据校验或训练失败时，给出简短中文错误。
    try:
        # 先确认数据完整，再开始提取特征。
        data = download_dataset(PROJECT_DIR / 'data/dermamnist_64.npz', offline=args.offline)
        # 结果目录由模型类型和评估模式共同决定。
        run_experiment(data, PROJECT_DIR / 'outputs' / experiment_folder / folder,
                       # 向实验函数传入是否评估test、是否训练RF两个开关。
                       args.evaluate_test, include_random_forest=args.random_forest)
    # 这里只处理可以向使用者直接说明的运行错误。
    except (OSError, ValueError, RuntimeError) as error:
        # 具体失败原因写到stderr。
        print(f'实验未完成：{error}', file=sys.stderr)
        # 以非零状态结束，便于脚本识别失败。
        return 1
    # 实验结束且没有异常时返回0。
    return 0


# 直接运行文件时才启动训练。
if __name__ == '__main__':
    # 把训练是否成功传回终端。
    sys.exit(main())
