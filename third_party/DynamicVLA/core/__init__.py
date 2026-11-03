# -*- coding: utf-8 -*-
#
# @File:   __init__.py
# @Author: Haozhe Xie
# @Date:   2025-05-15 20:06:15
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-15 20:06:18
# @Email:  root@haozhexie.com

def train(*args, **kwargs):
    """Keep the legacy entry point without eagerly importing video backends."""
    from .train import train as implementation
    globals()["train"] = _train_entry
    return implementation(*args, **kwargs)


def test(*args, **kwargs):
    from .test import test as implementation
    globals()["test"] = _test_entry
    return implementation(*args, **kwargs)


_train_entry, _test_entry = train, test
