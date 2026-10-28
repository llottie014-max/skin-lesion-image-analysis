"""像素级精确重复审计；不能替代病灶/患者身份核查。"""
from collections import defaultdict
import hashlib
from itertools import combinations

import numpy as np


def audit_splits(splits):
    """输入 {分区名: (图像, 标签)}；分组单位是完全相同的RGB像素。"""
    all_groups = defaultdict(list)
    within = {}
    sizes = {}
    for split, (images, labels) in splits.items():
        images, labels = np.asarray(images), np.asarray(labels).reshape(-1)
        if (images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8
                or len(images) != len(labels) or not len(images)):
            raise ValueError('审计需要数量匹配的非空RGB图像与标签。')
        if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 6)):
            raise ValueError('审计标签必须为0～6整数。')
        sizes[split] = len(images)
        local = defaultdict(list)
        for index, (image, label) in enumerate(zip(images, labels)):
            payload = str(image.shape).encode('ascii') + image.tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            member = {'split': split, 'image_index': index, 'label': int(label)}
            local[digest].append(member)
            all_groups[digest].append(member)
        duplicates = [{'sha256': digest, 'members': members}
                      for digest, members in local.items() if len(members) > 1]
        within[split] = {
            'duplicate_groups': len(duplicates),
            'extra_images': sum(len(group['members']) - 1 for group in duplicates),
            'groups': duplicates,
        }
    shared, conflicts = [], []
    pairs = {'__'.join(pair): 0 for pair in combinations(splits, 2)}
    for digest, members in all_groups.items():
        present = {member['split'] for member in members}
        if len(present) > 1:
            shared.append({'sha256': digest, 'members': members})
            for first, second in combinations(splits, 2):
                if {first, second}.issubset(present):
                    pairs[f'{first}__{second}'] += 1
        if len({member['label'] for member in members}) > 1:
            conflicts.append({'sha256': digest, 'members': members})
    return {
        'method': 'SHA256 of image shape and exact uint8 RGB pixels; no near-duplicate or lesion identity matching',
        'split_sizes': sizes, 'within_split': within,
        'cross_split_shared_groups': len(shared), 'cross_split_pair_shared_groups': pairs,
        'conflicting_label_groups': len(conflicts),
        'shared_groups': shared, 'label_conflicts': conflicts,
        'limitation': 'Zero exact duplicates does not establish lesion or patient independence. Original DermaMNIST splits have reported same-lesion overlap; no correction is performed here.',
    }
