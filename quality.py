"""像素级精确重复审计；不能替代病灶/患者身份核查。"""
# 同一个哈希下面可以直接追加多个图像记录。
from collections import defaultdict
# 对图像像素计算SHA256摘要。
import hashlib
# 枚举train、val、test之间的两两组合。
from itertools import combinations

# 检查数组类型和标签范围。
import numpy as np


def audit_splits(splits):
    """输入 {分区名: (图像, 标签)}；分组单位是完全相同的RGB像素。"""
    # 保存所有分区的哈希分组，用来查跨分区重复。
    all_groups = defaultdict(list)
    # 分别记录每个分区内部的重复。
    within = {}
    # 记录参与审计的图像数量。
    sizes = {}
    # 依次处理每个分区及其图像、标签。
    for split, (images, labels) in splits.items():
        # 标签展平成一维，与图像逐张配对。
        images, labels = np.asarray(images), np.asarray(labels).reshape(-1)
        # 原始像素必须是uint8 RGB数组。
        if (images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8
                # 图像与标签数量要一致，并且分区不能为空。
                or len(images) != len(labels) or not len(images)):
            # 格式不符时停止，避免得到误导性的重复统计。
            raise ValueError('审计需要数量匹配的非空RGB图像与标签。')
        # 七个类别只允许整数0到6。
        if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 6)):
            # 无效标签不能进入冲突检查。
            raise ValueError('审计标签必须为0～6整数。')
        # 保留分区大小，核对审计覆盖范围。
        sizes[split] = len(images)
        # local只记录当前分区，和all_groups分开统计。
        local = defaultdict(list)
        # 同时取出图像编号、像素和类别。
        for index, (image, label) in enumerate(zip(images, labels)):
            # 把形状也放进摘要输入，区分不同形状的相同字节串。
            payload = str(image.shape).encode('ascii') + image.tobytes()
            # 相同形状与像素会得到相同的摘要。
            digest = hashlib.sha256(payload).hexdigest()
            # 保留位置和标签，找到重复后能追溯原图。
            member = {'split': split, 'image_index': index, 'label': int(label)}
            # 当前分区中按摘要归组。
            local[digest].append(member)
            # 所有分区中也按同一摘要归组。
            all_groups[digest].append(member)
        # 一个摘要有多张图像时，记作一个重复组。
        duplicates = [{'sha256': digest, 'members': members}
                      # 只保留成员数大于1的组。
                      for digest, members in local.items() if len(members) > 1]
        # 下面同时记录组数、额外图像数和明细。
        within[split] = {
            # 一组重复可能有三张以上图像，组数不等于图像数。
            'duplicate_groups': len(duplicates),
            # 每组保留一张后，多出来的图像数是成员数减1。
            'extra_images': sum(len(group['members']) - 1 for group in duplicates),
            # 成员明细用于人工核查。
            'groups': duplicates,
        }
    # 跨分区共享和标签冲突分别统计，二者不一定同时发生。
    shared, conflicts = [], []
    # 初始化train-val、train-test、val-test三个组合的计数。
    pairs = {'__'.join(pair): 0 for pair in combinations(splits, 2)}
    # 逐个检查所有摘要组。
    for digest, members in all_groups.items():
        # 用集合找出这组像素出现在哪些分区。
        present = {member['split'] for member in members}
        # 出现在两个以上分区，就属于跨分区共享。
        if len(present) > 1:
            # 保存整组成员，便于定位具体图像。
            shared.append({'sha256': digest, 'members': members})
            # 对每对分区检查一次。
            for first, second in combinations(splits, 2):
                # 只有两边都出现该摘要时才计入这对分区。
                if {first, second}.issubset(present):
                    # 增加的是共享组数，不是所有图像配对数。
                    pairs[f'{first}__{second}'] += 1
        # 同样像素却有多个类别编号，说明标签存在冲突。
        if len({member['label'] for member in members}) > 1:
            # 标签冲突明细单独保存。
            conflicts.append({'sha256': digest, 'members': members})
    # 返回可直接序列化的审计报告。
    return {
        # 说明哈希只检查精确像素，不识别相似照片或病灶身份。
        'method': 'SHA256 of image shape and exact uint8 RGB pixels; no near-duplicate or lesion identity matching',
        # 分区大小和内部重复结果放在一起。
        'split_sizes': sizes, 'within_split': within,
        # 给出跨分区总数，以及两两分区的共享组数。
        'cross_split_shared_groups': len(shared), 'cross_split_pair_shared_groups': pairs,
        # 冲突组数用于快速检查是否存在矛盾标签。
        'conflicting_label_groups': len(conflicts),
        # 保留共享组、冲突组的逐张图像位置。
        'shared_groups': shared, 'label_conflicts': conflicts,
        # 零精确重复也不能证明病灶或患者相互独立。
        'limitation': 'Zero exact duplicates does not establish lesion or patient independence. Original DermaMNIST splits have reported same-lesion overlap; no correction is performed here.',
    }
