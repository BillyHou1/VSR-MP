# Fan
# Generate JSON file lists for VoxCeleb2.
# dev set is for training, test set stays as-is.
# Hold out ~2-3% of dev as validation, use a fixed seed.
# Same format as LRS2: [{"video": "/abs/path/to/clip.mp4"}, ...]
# Output: data/vox_train.json, data/vox_valid.json, data/vox_test.json

import os
import json
import random
import argparse


def collect_clips(vox2_root, subset='dev'):
    """
    Walk a VoxCeleb2 subset directory and collect all .mp4 clip paths.

    Args:
        vox2_root: str, root directory of VoxCeleb2
        subset:    str, 'dev' or 'test'
    Returns:
        list of {"video": "/abs/path/to/clip.mp4"}
    """
    # TODO
    raise NotImplementedError


def split_dev(dev_list, val_ratio=0.03, seed=1234):
    """
    Hold out a small portion of dev as validation.

    Args:
        dev_list:  list of dicts from collect_clips()
        val_ratio: float, fraction to hold out
        seed:      int, for reproducibility
    Returns:
        (train_list, valid_list)
    """
    # TODO
    raise NotImplementedError


def save_json(data, path):
    """Save a list to a JSON file."""
    # TODO
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Generate VoxCeleb2 JSON lists')
    parser.add_argument('--vox2_root', required=True, help='root directory of VoxCeleb2')
    parser.add_argument('--output_dir', default='data', help='output directory for JSON files')
    args = parser.parse_args()

    # TODO collect clips, split dev, save JSONs, print counts
    raise NotImplementedError


if __name__ == '__main__':
    main()
