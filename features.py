"""从RGB图像提取24维可解释特征；不使用标签，也不拟合数据。"""
import numpy as np

from workflow import to_gray


FEATURE_NAMES = (
    ['red_mean', 'green_mean', 'blue_mean', 'red_std', 'green_std', 'blue_std',
     'gray_mean', 'gray_std']
    + [f'gray_hist_{index:02d}' for index in range(16)]
)


def extract_features(images):
    """颜色与亮度统计缩放到0～1；直方图16格，每行概率之和为1。"""
    images = np.asarray(images)
    if (images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8
            or images.shape[0] == 0 or min(images.shape[1:3]) == 0):
        raise ValueError('需要非空的uint8 RGB图像数组 (N,H,W,3)。')
    rows = []
    for image in images:
        rgb = image.astype(np.float64) / 255.0
        gray = to_gray(image)
        histogram, _ = np.histogram(gray, bins=16, range=(0, 256))
        rows.append(np.concatenate([
            rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)),
            [gray.mean() / 255.0, gray.std() / 255.0],
            histogram / gray.size,
        ]))
    return np.asarray(rows, dtype=np.float64)
