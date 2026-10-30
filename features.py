"""从RGB图像提取24维可解释特征；不使用标签，也不拟合数据。"""
# 用NumPy计算均值、标准差和直方图。
import numpy as np

# 复用同一种灰度转换，避免分析和训练使用不同定义。
from workflow import to_gray


# 特征名的顺序必须和下面拼接数值的顺序相同。
FEATURE_NAMES = (
    # 先放RGB均值，再放RGB标准差。
    ['red_mean', 'green_mean', 'blue_mean', 'red_std', 'green_std', 'blue_std',
     # 接着放灰度均值和灰度标准差。
     'gray_mean', 'gray_std']
    # 最后16列是灰度直方图各区间的像素比例。
    + [f'gray_hist_{index:02d}' for index in range(16)]
)


def extract_features(images):
    """颜色与亮度统计缩放到0～1；直方图16格，每行概率之和为1。"""
    # 统一成NumPy数组，方便检查形状。
    images = np.asarray(images)
    # 要求四维数组，最后一维为RGB三个通道。
    if (images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8
            # 图像数量、高度和宽度都不能为0。
            or images.shape[0] == 0 or min(images.shape[1:3]) == 0):
        # uint8对应原始0到255像素，浮点图像不在这里自动猜测范围。
        raise ValueError('需要非空的uint8 RGB图像数组 (N,H,W,3)。')
    # 一张图像生成一行，共24个特征。
    rows = []
    # 逐张处理，不读取标签，也不学习其他图像的信息。
    for image in images:
        # 先转浮点数，再把RGB像素缩放到0到1。
        rgb = image.astype(np.float64) / 255.0
        # Pillow按亮度权重转换灰度，并非直接平均RGB。
        gray = to_gray(image)
        # 固定16个区间，每格宽16，覆盖0到255所有灰度值。
        histogram, _ = np.histogram(gray, bins=16, range=(0, 256))
        # 将不同类型的统计量接成一个一维向量。
        rows.append(np.concatenate([
            # 沿高度和宽度求统计量，保留三个颜色通道。
            rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)),
            # 灰度均值与标准差也除以255，统一数值范围。
            [gray.mean() / 255.0, gray.std() / 255.0],
            # 除以像素总数，得到各区间比例，而不是像素个数。
            histogram / gray.size,
        ]))
    # 最终形状为(N,24)，每行对应输入中的同一张图像。
    return np.asarray(rows, dtype=np.float64)
