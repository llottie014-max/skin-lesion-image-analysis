"""皮肤病变图像分析：按下载、读取、统计、绘图的顺序阅读。"""

# 用MD5检查下载文件是否与官方文件一致。
import hashlib
# 把数据来源和输出信息保存成JSON。
import json
# 用Path组合路径，兼容不同系统的分隔符。
from pathlib import Path
# 通过Python标准库下载数据，不依赖额外下载工具。
import urllib.request

# 先设置绘图后端，再导入pyplot。
import matplotlib
matplotlib.use('Agg')  # 保存图片，无需弹出图形窗口。
# 用于柱状图、图像网格和直方图。
import matplotlib.pyplot as plt
# 检查本机有没有可用的中文字体。
from matplotlib import font_manager
# 用数组处理图像、标签和像素统计。
import numpy as np
# 将统计结果写成CSV表格。
import pandas as pd
# 用Pillow读取数组图像、转换灰度和缩放。
from PIL import Image



# 所有相对路径从脚本目录出发，避免依赖终端当前位置。
PROJECT_DIR = Path(__file__).resolve().parent
# 固定使用官方发布的64×64版本。
DATA_URL = 'https://zenodo.org/records/10519652/files/dermamnist_64.npz?download=1'
# 官方文件的MD5值，用于识别损坏或错误文件。
DATA_MD5 = 'b70a2f5635c6199aeaa28c31d7202e1f'
# 保留官方分区大小，不自行重新切分。
SPLIT_COUNTS = {'train': 7007, 'val': 1003, 'test': 2005}
# 英文全称按标签0到6排列。
CLASS_NAMES = [
    # 类别0的英文名称。
    'actinic keratoses and intraepithelial carcinoma',
    # 依次对应类别1和类别2。
    'basal cell carcinoma', 'benign keratosis-like lesions',
    # 依次对应类别3到类别6。
    'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesions',
]
# 中文名称与英文名称、类别编号一一对应。
CLASS_ZH = [
    # 依次对应类别0、1、2。
    '日光性角化病及上皮内癌', '基底细胞癌', '良性角化样病变',
    # 依次对应类别3、4、5、6。
    '皮肤纤维瘤', '黑色素瘤', '黑色素细胞痣', '血管性病变',
]
# 图表使用短名称，避免横轴过于拥挤。
CLASS_SHORT = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
# 每个类别固定一种颜色，便于在不同图里对照。
COLORS = ['#4c78a8', '#f58518', '#e45756', '#72b7b2', '#54a24b', '#b279a2', '#ff9da6']


def check_md5(path, expected):
    """分块校验文件，防止不完整下载或损坏文件参与分析。"""
    # 建立摘要对象，后面分块更新。
    digest = hashlib.md5()
    # 用二进制方式读取，不能把压缩文件当成文字读。
    with Path(path).open('rb') as stream:
        # 每次读1MiB，读到空字节串时结束，避免一次占用全部内存。
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            # 按原文件顺序累积每块内容。
            digest.update(chunk)
    # 把最终摘要和预期值比较。
    if digest.hexdigest() != expected:
        # 校验失败时保留原文件，提示使用者移走后再下载。
        raise ValueError(f'数据校验失败：{path}。请移走该损坏文件后重新联网下载。')


def download_dataset(path, offline=False):
    """正式文件存在时只校验，不访问网络；新下载先写临时文件。"""
    # 允许调用者传字符串路径，内部统一转Path。
    path = Path(path)
    # 已有文件时优先复用缓存。
    if path.exists():
        # 缓存也要校验，不能仅凭文件存在就信任它。
        check_md5(path, DATA_MD5)
        # 说明本次使用本地数据，没有重新下载。
        print('  已找到并校验本地缓存。', flush=True)
        # 返回通过校验的文件位置。
        return path
    # 离线模式下禁止尝试下载。
    if offline:
        # 缺少缓存时明确告诉使用者先联网运行。
        raise FileNotFoundError(f'离线模式下找不到数据：{path}；请先联网运行一次。')
    # 缺少父目录时一并创建，已有目录可复用。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 下载中的文件加.part后缀，避免被当成完整数据。
    temporary = path.with_suffix('.npz.part')
    # 下载前显示一次提示。
    print('  首次下载约 100 MB，请稍候……', flush=True)
    # 下载失败也会执行finally，清理残留临时文件。
    try:
        # 设置请求头，让服务器识别这是课程项目的下载请求。
        request = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Python-course-image-analysis/1.0'})
        # 设置网络超时，并将响应以二进制写入临时文件。
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
            # 累计收到的字节数，从0开始。
            received = 0
            # 每收到约10MiB输出一次进度。
            next_notice = 10 * 1024 * 1024
            # 持续读数据，直到服务器没有新内容。
            while True:
                # 每次最多读取1MiB。
                chunk = response.read(1024 * 1024)
                # 空字节串表示响应已结束。
                if not chunk:
                    # 跳出下载循环。
                    break
                # 把这一块原样写入文件。
                output.write(chunk)
                # 按实际块大小更新下载量，最后一块可能不足1MiB。
                received += len(chunk)
                # 达到下一次提示阈值才打印进度。
                if received >= next_notice:
                    # 把字节换成MiB显示，保留整数。
                    print(f'  已下载 {received / 1024**2:.0f} MiB', flush=True)
                    # 将下一次提示阈值再增加10MiB。
                    next_notice += 10 * 1024 * 1024
        # 先校验下载完成的临时文件。
        check_md5(temporary, DATA_MD5)
        # 校验通过后，才将它改成正式文件名。
        temporary.replace(path)
    # 无论下载成功还是抛出异常，都会执行清理。
    finally:
        # 改名后临时文件已不存在，missing_ok允许这种情况。
        temporary.unlink(missing_ok=True)
    # 返回正式文件位置，供读取函数使用。
    return path


def load_dataset(path, split='train'):
    """读取指定分区；图像为 N×64×64×3，标签展平为 N 个整数。"""
    # 先检查分区名，防止拼错键名。
    if split not in SPLIT_COUNTS:
        # 可用分区限定为官方的三个分区。
        raise ValueError('split 必须为 train、val 或 test。')
    # 禁用pickle，读取数据不需要执行Python对象。
    with np.load(path, allow_pickle=False) as dataset:
        # NPZ中每个分区分别保存图像和标签两个数组。
        image_key, label_key = f'{split}_images', f'{split}_labels'
        # 两个数组必须同时存在。
        if image_key not in dataset or label_key not in dataset:
            # 错误里指出缺失的键名，方便核对数据文件。
            raise ValueError(f'数据缺少 {image_key} 或 {label_key}。')
        # 取出原始RGB图像数组。
        images = dataset[image_key]
        # 标签先保留原形状，完成检查后再展平。
        raw_labels = dataset[label_key]
    # 要求非空的四维数组，每张图像都是64×64×3。
    if images.ndim != 4 or images.shape[1:] != (64, 64, 3) or len(images) == 0:
        # 形状错误时立即停止，不能继续假定它是官方图像。
        raise ValueError('图像必须为非空的 (N,64,64,3) RGB 数组。')
    # uint8表示每个通道像素值为0到255的整数。
    if images.dtype != np.uint8:
        # 不自动转换类型，避免掩盖错误数据。
        raise ValueError('图像像素应为 uint8 类型。')
    # 接受(N,)和(N,1)两种标签形状。
    if raw_labels.shape not in ((len(images),), (len(images), 1)):
        # 标签数量或维度不对时不能与图像配对。
        raise ValueError('标签形状或数量与图像不匹配。')
    # 浮点类别编号不按整数悄悄截断。
    if not np.issubdtype(raw_labels.dtype, np.integer):
        # 明确要求原始类别为整数。
        raise ValueError('标签必须为整数。')
    # reshape(-1)把(N,1)标签变成(N,)。
    labels = raw_labels.reshape(-1)
    # 逐项检查有没有小于0或大于6的类别编号。
    if np.any((labels < 0) | (labels > 6)):
        # 不属于七类的数据不能进入后续统计。
        raise ValueError('标签必须在 0 到 6 之间。')
    # 返回图像数组和一维标签，图像像素保持原样。
    return images, labels


def class_statistics(labels):
    """统计七类图像数量；比例是数据集图像比例，不是患者患病率。"""
    # 统计0到6各出现多少次，缺少的类也保留0。
    counts = np.bincount(labels, minlength=7)
    # 每个类别占表格的一行。
    return pd.DataFrame({
        # 同一行保存类别编号和中英文名称。
        'label': range(7), 'class_name': CLASS_NAMES, 'class_name_zh': CLASS_ZH,
        # 比例的分母是当前分区图像总数。
        'count': counts, 'percent': counts / len(labels) * 100,
    })


def to_gray(image):
    """RGB 转亮度灰度，输出二维 uint8 数组，像素范围 0～255。"""
    # L模式按RGB亮度权重转灰度，再转回NumPy数组。
    return np.asarray(Image.fromarray(image).convert('L'))


def configure_font():
    """优先选择本机中文字体；没有时使用英文图表。"""
    # 收集Matplotlib实际发现的字体名称。
    available = {font.name for font in font_manager.fontManager.ttflist}
    # 按优先顺序寻找中文字体。
    for name in ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'Noto Sans CJK SC', 'SimHei']:
        # 找到候选字体时再设置，避免不存在字体的警告。
        if name in available:
            # 中文用所选字体，缺失字符回退到DejaVu Sans。
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            # 避免某些中文字体将坐标轴负号显示成方框。
            plt.rcParams['axes.unicode_minus'] = False
            # 调用者据此选择中文标题。
            return True
    # 没有中文字体时，调用者使用英文标题。
    return False


def save_figure(fig, path, split):
    # 在图外下边缘注明数据集、分区和教学用途。
    fig.text(0.99, -0.014, f'DermaMNIST | {split} | educational image analysis',
             # 文字靠右对齐，使用较小的灰色字体。
             ha='right', fontsize=8, color='#555555')
    # 以300dpi保存，tight把图外标题和脚注一并纳入。
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    # 每张图保存后关闭，避免连续绘图占用太多内存。
    plt.close(fig)


def run_analysis(data_path, output_dir, split='train', image_index=0):
    """完整分析一个分区，生成真实图像统计和可复查输出。"""
    # 读取所选分区，并完成图像和标签格式检查。
    images, labels = load_dataset(data_path, split)
    # 示例编号必须是当前分区内的整数下标。
    if not isinstance(image_index, int) or not 0 <= image_index < len(images):
        # 报错时给出实际可选范围。
        raise ValueError(f'--image-index 必须为 0～{len(images) - 1} 之间的整数。')
    # 核对是否真的是完整的官方分区。
    if len(images) != SPLIT_COUNTS[split]:
        # 数量不符时不输出看似完整的统计结果。
        raise ValueError(f'官方 {split} 分区应为 {SPLIT_COUNTS[split]} 张，实际 {len(images)} 张。')
    # 统一结果目录的路径类型。
    output_dir = Path(output_dir)
    # 创建结果目录，允许覆盖同名的本次分析输出。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 记录中文字体是否可用，后面据此选标题。
    chinese = configure_font()
    # 打印数组形状与像素类型，便于确认读取结果。
    print(f'[2/5] 读取 {split}：{images.shape}，像素类型 {images.dtype}', flush=True)
    # 按标签计算每类图像数和比例。
    statistics = class_statistics(labels)
    # 不写额外行号，使用带BOM的UTF-8便于Excel识别中文。
    statistics.to_csv(output_dir / 'class_statistics.csv', index=False, encoding='utf-8-sig')
    # 终端中只显示编号、中文名和数量，便于快速核对。
    print(statistics[['label', 'class_name_zh', 'count']].to_string(index=False), flush=True)

    # 提示接下来开始逐图计算特征。
    print('[3/5] 提取逐图亮度和颜色特征……', flush=True)
    # 每张图像的特征暂存在一个字典里。
    records = []
    # enumerate给图像附上可追溯的分区内编号。
    for index, (image, label) in enumerate(zip(images, labels)):
        # 同一张图像转灰度后计算亮度特征。
        gray = to_gray(image)
        # 沿高度和宽度求均值，得到R、G、B三个数。
        rgb_mean = image.mean(axis=(0, 1))
        # 一张图像对应一条记录。
        records.append({
            # 保留分区、编号和原有标签，便于定位原图。
            'split': split, 'image_index': index, 'label': int(label),
            # 根据标签编号查出英文类别名。
            'class_name': CLASS_NAMES[int(label)],
            # 红、绿通道均值保持原始0到255量纲。
            'red_mean': float(rgb_mean[0]), 'green_mean': float(rgb_mean[1]),
            # 记录蓝通道均值和整张灰度图的平均亮度。
            'blue_mean': float(rgb_mean[2]), 'gray_mean': float(gray.mean()),
            # 灰度标准差反映这张图像亮度的离散程度。
            'gray_std': float(gray.std()),
        })
    # 把逐图记录写成表格，不额外写DataFrame行号。
    pd.DataFrame(records).to_csv(output_dir / 'image_features.csv', index=False,
                                 # 浮点数保留4位小数，便于阅读和复查。
                                 encoding='utf-8-sig', float_format='%.4f')

    # 接下来生成四类图表，输入均来自当前分区。
    print('[4/5] 绘制类别统计、样本、处理对比和直方图……', flush=True)
    # 建立单张柱状图，自动给标题和坐标轴留出空间。
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    # 横轴是类别缩写，柱高是实际图像数量。
    bars = ax.bar(CLASS_SHORT, statistics['count'], color=COLORS)
    # 直接在柱顶标出数量。
    ax.bar_label(bars, padding=3)
    # 纵轴上限多留16%，防止柱顶数字被裁掉。
    ax.set_ylim(0, max(statistics['count']) * 1.16)
    # 标题包含分区和样本总数，方便区分不同输出。
    ax.set(title=f'皮肤病变图像类别分布（{split}，n={len(images)}）' if chinese else f'Class distribution ({split}, n={len(images)})',
           # 按字体是否可用选择横轴语言。
           xlabel='类别（缩写对应样本图）' if chinese else 'Class',
           # 纵轴表示图像张数。
           ylabel='图像数量' if chinese else 'Number of images')
    # 网格线放在柱子下面，减少遮挡。
    ax.set_axisbelow(True)
    # 只画淡色横向网格，便于比较柱高。
    ax.grid(axis='y', alpha=0.25)
    # 保存类别分布图并关闭画布。
    save_figure(fig, output_dir / 'class_distribution.png', split)

    # 固定随机种子，使重复运行选出相同的示例。
    rng = np.random.default_rng(42)
    # 七行对应七类，每类最多展示三张。
    fig, axes = plt.subplots(7, 3, figsize=(9, 18), layout='constrained')
    # 遍历每一类对应的一行子图。
    for label, row in enumerate(axes):
        # 找出当前类别所有图像的下标。
        candidates = np.flatnonzero(labels == label)
        # 无放回抽样，同类图像不足三张时取实际数量。
        selected = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
        # 遍历这一行的三个位置。
        for column, ax in enumerate(row):
            # 只有确实抽到样本的位置才显示图像。
            if column < len(selected):
                # 将NumPy整数转成普通Python整数。
                index = int(selected[column])
                # 直接显示原始RGB像素。
                ax.imshow(images[index])
                # 中文字体可用时展示中文类名。
                name = CLASS_ZH[label] if chinese else CLASS_SHORT[label]
                # 写上类别和原图编号，便于回到数据里查找。
                ax.set_title(f'{CLASS_SHORT[label]}: {name}\nindex={index}', fontsize=10)
            # 这一类样本不足三个时，剩余位置显示提示。
            else:
                # 在空白格中央标明没有样本。
                ax.text(0.5, 0.5, 'No sample', ha='center')
            # 隐藏图像周围的坐标轴刻度。
            ax.axis('off')
    # 强调这里展示的是数据已有标签。
    fig.suptitle(f'各类别样本（已有标签，{split}）' if chinese else f'Example images (existing labels, {split})')
    # 保存随机抽样的类别样本图。
    save_figure(fig, output_dir / 'sample_gallery.png', split)

    # first在这里指用户选中的示例，编号由image_index决定。
    first = images[image_index]
    # 后续处理图和直方图都使用这同一张示例。
    gray = to_gray(first)
    # 额外保留分区第0张图，方便最基础的读取练习。
    Image.fromarray(images[0]).save(output_dir / 'first_image.png')
    # 将用户选中的示例另存，避免与第0张混淆。
    Image.fromarray(first).save(output_dir / 'example_image.png')
    # 双线性插值缩小到32×32，仅演示处理，不用于训练。
    resized = np.asarray(Image.fromarray(first).resize((32, 32), Image.Resampling.BILINEAR))
    # 并排比较原图、缩放图和灰度图。
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), layout='constrained')
    # 三个标题的顺序与下面三个数组的顺序保持一致。
    titles = ['原图 64×64', '缩放后 32×32', '灰度图 64×64'] if chinese else ['Original 64x64', 'Resized 32x32', 'Grayscale 64x64']
    # 将每个子图、图像和标题一一配对。
    for ax, array, title in zip(axes, [first, resized, gray], titles):
        # 二维图像用灰度色图，RGB图像保留原色。
        ax.imshow(array, cmap='gray' if array.ndim == 2 else None, vmin=0, vmax=255, interpolation='nearest')
        # 为当前处理结果设置标题。
        ax.set_title(title)
        # 图像比较不需要坐标刻度。
        ax.axis('off')
    # 总标题注明示例编号和类别，强调来自同一图像。
    fig.suptitle(f'{split} index={image_index} | {CLASS_SHORT[int(labels[image_index])]} | same image')
    # 保存处理前后对比图。
    save_figure(fig, output_dir / 'processing_comparison.png', split)

    # 左边画RGB直方图，右边画灰度直方图。
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    # 257个边界把0到255分成256个像素强度区间。
    bins = np.arange(257)
    # 每个通道使用与其名称相符的颜色。
    for channel, name, color in [(0, 'Red', '#d62728'), (1, 'Green', '#2ca02c'), (2, 'Blue', '#1f77b4')]:
        # 先展平一个颜色通道，再按像素值统计频数。
        axes[0].hist(first[:, :, channel].ravel(), bins=bins, histtype='step', label=name, color=color)
    # 图例标明三条线分别属于哪个通道。
    axes[0].legend()
    # 为RGB直方图设置标题。
    axes[0].set_title('RGB histogram')
    # 灰度图展平成一维后，使用相同的区间边界统计。
    axes[1].hist(gray.ravel(), bins=bins, color='#666666')
    # 右侧标题说明这是灰度直方图。
    axes[1].set_title('Grayscale histogram')
    # 统一两个子图的坐标轴设置。
    for ax in axes:
        # 横轴是像素值，纵轴是该像素值出现的次数。
        ax.set(xlabel='Pixel intensity (0-255)', ylabel='Pixel count', xlim=(0, 255))
        # 加浅色网格，方便读出频数。
        ax.grid(alpha=0.2)
    # 再次标明两个直方图来自同一张示例。
    fig.suptitle(f'{split} index={image_index} | RGB and grayscale of the same image')
    # 保存像素分布图。
    save_figure(fig, output_dir / 'intensity_histogram.png', split)

    # 把本次分析的输入、参数和输出清单集中记录。
    summary = {
        # 数据集名称、下载地址和校验值用于核对来源。
        'dataset': 'DermaMNIST', 'source_url': DATA_URL, 'md5': DATA_MD5,
        # 记录许可、当前分区和实际图像数量。
        'license': 'CC BY-NC 4.0', 'split': split, 'image_count': len(images),
        # 数组形状转为列表，数据类型转为字符串，便于写JSON。
        'image_shape': list(images.shape), 'dtype': str(images.dtype),
        # 保留七类数量和抽样随机种子。
        'class_counts': statistics['count'].tolist(), 'gallery_seed': 42,
        # 记录特征表行数与用户选中的示例编号。
        'feature_rows': len(records), 'example_index': image_index,
        # 说明这些是图像统计，不能当成患病率或模型预测。
        'note': 'Existing dataset labels; image statistics only, not patient prevalence or model predictions. Original splits have reported lesion overlap.',
        # 输出清单先列出统计表、特征表和第0张图。
        'artifacts': ['class_statistics.csv', 'image_features.csv', 'first_image.png',
                      # 清单继续列出类别分布和样本图。
                      'class_distribution.png', 'sample_gallery.png',
                      # 列出处理对比图与直方图。
                      'processing_comparison.png', 'intensity_histogram.png',
                      # 最后列出用户指定的示例原图。
                      'example_image.png'],
    }
    # 保留中文和缩进，文件末尾补换行，方便直接阅读。
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # 打印绝对结果路径，便于在文件管理器里打开。
    print(f'[5/5] 完成，结果保存在：{output_dir}', flush=True)
    # 返回摘要，供测试或其他Python脚本检查。
    return summary
