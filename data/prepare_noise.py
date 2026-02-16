# Fan
# Scan noise directories for audio files, merge into one pool,
# split train/valid (~5% held out), save as JSON lists.
# Output: data/noise_train.json, data/noise_valid.json
# Each is a flat list of absolute paths.

import os
import json
import random
import argparse


def scan_noise_dirs(noise_dirs, extensions=('.wav', '.flac')):
    """
    Recursively scan directories for audio files.

    Args:
        noise_dirs:  list of str, directories to scan
        extensions:  tuple of str, file extensions to include
    Returns:
        list of absolute file paths
    """
    # TODO
    raise NotImplementedError


def split_noise(file_list, val_ratio=0.05, seed=1234):
    """
    Split noise files into train and validation sets.

    Args:
        file_list: list of str, file paths
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
    parser = argparse.ArgumentParser(description='Prepare noise pool from multiple sources')
    parser.add_argument('--noise_dirs', nargs='+', required=True,
                        help='directories containing noise audio files')
    parser.add_argument('--output_dir', default='data', help='output directory for JSON files')
    args = parser.parse_args()

    # TODO scan, split, save, print counts
    raise NotImplementedError


if __name__ == '__main__':
    main()
