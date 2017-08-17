#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def sim_reg_data(n_samples=50, n_features=200, seed=42,
                        n_sparse_features=10):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    w = 3 * np.random.randn(n_features)
    inds = np.arange(n_features)
    np.random.shuffle(inds)
    # y = Xw + e
    y = np.dot(X, w)
    y += 0.01 * np.random.normal((n_samples,))

    return X, y


def sim_reg_sparse_data(n_samples=50, n_features=200, seed=42,
                        n_sparse_features=10):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    w = 3 * np.random.randn(n_features)
    inds = np.arange(n_features)
    np.random.shuffle(inds)
    w[inds[n_sparse_features:]] = 0  # sparsify coef
    # y = Xw + e
    y = np.dot(X, w)
    y += 0.01 * np.random.normal((n_samples,))

    return X, y


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--data-file',
        default='',
        help='Files containing counts')
    options = parser.parse_args()
    print(options)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
