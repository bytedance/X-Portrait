# -*- coding: utf-8 -*-

import os
import errno
import pickle
from collections import deque
import torch


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def pickle_load(file_path):
    assert os.path.isfile(file_path), file_path
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file


def pickle_dump(file_path, file):
    if os.path.isfile(file_path):
        os.remove(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(file, f, protocol=2)


def walk_all_files_with_suffix(dir,
                               suffixs=('.jpg', '.png', '.jpeg'),
                               sort=False):
    paths = []
    names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            if os.path.splitext(fname)[1].lower() in suffixs:
                paths.append(path)
                if not sort:
                    names.append(fname)

    if sort:
        paths.sort()
        for i in range(len(paths)):
            names.append(os.path.basename(paths[i]))
    return len(names), names, paths


def get_dirs(root_dir):
    dir_paths = []
    dir_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            dir_paths.append(path)
            dir_names.append(os.path.basename(path))
    dir_names.sort()
    dir_paths.sort()
    return len(dir_names), dir_names, dir_paths


def get_leave_dirs(root_dir):
    leave_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames: 
            leave_dirs.append(dirpath)
    return leave_dirs


def merge_pkl_dict(paths):
    dicts = []
    for i in range(len(paths)):
        dicts.append(pickle_load(paths[i]))
    for i in range(len(paths) - 1):
        dicts[0].update(dicts[i + 1])
    return dicts[0]
