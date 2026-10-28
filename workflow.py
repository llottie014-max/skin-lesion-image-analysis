"""皮肤病变图像分析：按下载、读取、统计、绘图的顺序阅读。"""

import hashlib
import json
from pathlib import Path
import urllib.request

import matplotlib
matplotlib.use('Agg')  # 保存图片，无需弹出图形窗口。
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image



PROJECT_DIR = Path(__file__).resolve().parent
DATA_URL = 'https://zenodo.org/records/10519652/files/dermamnist_64.npz?download=1'
DATA_MD5 = 'b70a2f5635c6199aeaa28c31d7202e1f'
SPLIT_COUNTS = {'train': 7007, 'val': 1003, 'test': 2005}
CLASS_NAMES = [
    'actinic keratoses and intraepithelial carcinoma',
    'basal cell carcinoma', 'benign keratosis-like lesions',
    'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesions',
]
CLASS_ZH = [
    '日光性角化病及上皮内癌', '基底细胞癌', '良性角化样病变',
    '皮肤纤维瘤', '黑色素瘤', '黑色素细胞痣', '血管性病变',
]
CLASS_SHORT = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
COLORS = ['#4c78a8', '#f58518', '#e45756', '#72b7b2', '#54a24b', '#b279a2', '#ff9da6']


def check_md5(path, expected):
    """分块校验文件，防止不完整下载或损坏文件参与分析。"""
    digest = hashlib.md5()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'数据校验失败：{path}。请移走该损坏文件后重新联网下载。')


def download_dataset(path, offline=False):
    """正式文件存在时只校验，不访问网络；新下载先写临时文件。"""
    path = Path(path)
    if path.exists():
        check_md5(path, DATA_MD5)
        print('  已找到并校验本地缓存。', flush=True)
        return path
    if offline:
        raise FileNotFoundError(f'离线模式下找不到数据：{path}；请先联网运行一次。')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.part')
    print('  首次下载约 100 MB，请稍候……', flush=True)
    try:
        request = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Python-course-image-analysis/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
            received = 0
            next_notice = 10 * 1024 * 1024
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if received >= next_notice:
                    print(f'  已下载 {received / 1024**2:.0f} MiB', flush=True)
                    next_notice += 10 * 1024 * 1024
        check_md5(temporary, DATA_MD5)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_dataset(path, split='train'):
    """读取指定分区；图像为 N×64×64×3，标签展平为 N 个整数。"""
    if split not in SPLIT_COUNTS:
        raise ValueError('split 必须为 train、val 或 test。')
    with np.load(path, allow_pickle=False) as dataset:
        image_key, label_key = f'{split}_images', f'{split}_labels'
        if image_key not in dataset or label_key not in dataset:
            raise ValueError(f'数据缺少 {image_key} 或 {label_key}。')
        images = dataset[image_key]
        raw_labels = dataset[label_key]
    if images.ndim != 4 or images.shape[1:] != (64, 64, 3) or len(images) == 0:
        raise ValueError('图像必须为非空的 (N,64,64,3) RGB 数组。')
    if images.dtype != np.uint8:
        raise ValueError('图像像素应为 uint8 类型。')
    if raw_labels.shape not in ((len(images),), (len(images), 1)):
        raise ValueError('标签形状或数量与图像不匹配。')
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError('标签必须为整数。')
    labels = raw_labels.reshape(-1)
    if np.any((labels < 0) | (labels > 6)):
        raise ValueError('标签必须在 0 到 6 之间。')
    return images, labels


def class_statistics(labels):
    """统计七类图像数量；比例是数据集图像比例，不是患者患病率。"""
    counts = np.bincount(labels, minlength=7)
    return pd.DataFrame({
        'label': range(7), 'class_name': CLASS_NAMES, 'class_name_zh': CLASS_ZH,
        'count': counts, 'percent': counts / len(labels) * 100,
    })


def to_gray(image):
    """RGB 转亮度灰度，输出二维 uint8 数组，像素范围 0～255。"""
    return np.asarray(Image.fromarray(image).convert('L'))


def configure_font():
    """优先选择本机中文字体；没有时使用英文图表。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'Noto Sans CJK SC', 'SimHei']:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return True
    return False


def save_figure(fig, path, split):
    fig.text(0.99, -0.014, f'DermaMNIST | {split} | educational image analysis',
             ha='right', fontsize=8, color='#555555')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run_analysis(data_path, output_dir, split='train', image_index=0):
    """完整分析一个分区，生成真实图像统计和可复查输出。"""
    images, labels = load_dataset(data_path, split)
    if not isinstance(image_index, int) or not 0 <= image_index < len(images):
        raise ValueError(f'--image-index 必须为 0～{len(images) - 1} 之间的整数。')
    if len(images) != SPLIT_COUNTS[split]:
        raise ValueError(f'官方 {split} 分区应为 {SPLIT_COUNTS[split]} 张，实际 {len(images)} 张。')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chinese = configure_font()
    print(f'[2/5] 读取 {split}：{images.shape}，像素类型 {images.dtype}', flush=True)
    statistics = class_statistics(labels)
    statistics.to_csv(output_dir / 'class_statistics.csv', index=False, encoding='utf-8-sig')
    print(statistics[['label', 'class_name_zh', 'count']].to_string(index=False), flush=True)

    print('[3/5] 提取逐图亮度和颜色特征……', flush=True)
    records = []
    for index, (image, label) in enumerate(zip(images, labels)):
        gray = to_gray(image)
        rgb_mean = image.mean(axis=(0, 1))
        records.append({
            'split': split, 'image_index': index, 'label': int(label),
            'class_name': CLASS_NAMES[int(label)],
            'red_mean': float(rgb_mean[0]), 'green_mean': float(rgb_mean[1]),
            'blue_mean': float(rgb_mean[2]), 'gray_mean': float(gray.mean()),
            'gray_std': float(gray.std()),
        })
    pd.DataFrame(records).to_csv(output_dir / 'image_features.csv', index=False,
                                 encoding='utf-8-sig', float_format='%.4f')

    print('[4/5] 绘制类别统计、样本、处理对比和直方图……', flush=True)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    bars = ax.bar(CLASS_SHORT, statistics['count'], color=COLORS)
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(statistics['count']) * 1.16)
    ax.set(title=f'皮肤病变图像类别分布（{split}，n={len(images)}）' if chinese else f'Class distribution ({split}, n={len(images)})',
           xlabel='类别（缩写对应样本图）' if chinese else 'Class',
           ylabel='图像数量' if chinese else 'Number of images')
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.25)
    save_figure(fig, output_dir / 'class_distribution.png', split)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(7, 3, figsize=(9, 18), layout='constrained')
    for label, row in enumerate(axes):
        candidates = np.flatnonzero(labels == label)
        selected = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
        for column, ax in enumerate(row):
            if column < len(selected):
                index = int(selected[column])
                ax.imshow(images[index])
                name = CLASS_ZH[label] if chinese else CLASS_SHORT[label]
                ax.set_title(f'{CLASS_SHORT[label]}: {name}\nindex={index}', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No sample', ha='center')
            ax.axis('off')
    fig.suptitle(f'各类别样本（已有标签，{split}）' if chinese else f'Example images (existing labels, {split})')
    save_figure(fig, output_dir / 'sample_gallery.png', split)

    first = images[image_index]
    gray = to_gray(first)
    Image.fromarray(images[0]).save(output_dir / 'first_image.png')
    Image.fromarray(first).save(output_dir / 'example_image.png')
    resized = np.asarray(Image.fromarray(first).resize((32, 32), Image.Resampling.BILINEAR))
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), layout='constrained')
    titles = ['原图 64×64', '缩放后 32×32', '灰度图 64×64'] if chinese else ['Original 64x64', 'Resized 32x32', 'Grayscale 64x64']
    for ax, array, title in zip(axes, [first, resized, gray], titles):
        ax.imshow(array, cmap='gray' if array.ndim == 2 else None, vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f'{split} index={image_index} | {CLASS_SHORT[int(labels[image_index])]} | same image')
    save_figure(fig, output_dir / 'processing_comparison.png', split)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    bins = np.arange(257)
    for channel, name, color in [(0, 'Red', '#d62728'), (1, 'Green', '#2ca02c'), (2, 'Blue', '#1f77b4')]:
        axes[0].hist(first[:, :, channel].ravel(), bins=bins, histtype='step', label=name, color=color)
    axes[0].legend()
    axes[0].set_title('RGB histogram')
    axes[1].hist(gray.ravel(), bins=bins, color='#666666')
    axes[1].set_title('Grayscale histogram')
    for ax in axes:
        ax.set(xlabel='Pixel intensity (0-255)', ylabel='Pixel count', xlim=(0, 255))
        ax.grid(alpha=0.2)
    fig.suptitle(f'{split} index={image_index} | RGB and grayscale of the same image')
    save_figure(fig, output_dir / 'intensity_histogram.png', split)

    summary = {
        'dataset': 'DermaMNIST', 'source_url': DATA_URL, 'md5': DATA_MD5,
        'license': 'CC BY-NC 4.0', 'split': split, 'image_count': len(images),
        'image_shape': list(images.shape), 'dtype': str(images.dtype),
        'class_counts': statistics['count'].tolist(), 'gallery_seed': 42,
        'feature_rows': len(records), 'example_index': image_index,
        'note': 'Existing dataset labels; image statistics only, not patient prevalence or model predictions. Original splits have reported lesion overlap.',
        'artifacts': ['class_statistics.csv', 'image_features.csv', 'first_image.png',
                      'class_distribution.png', 'sample_gallery.png',
                      'processing_comparison.png', 'intensity_histogram.png',
                      'example_image.png'],
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[5/5] 完成，结果保存在：{output_dir}', flush=True)
    return summary
