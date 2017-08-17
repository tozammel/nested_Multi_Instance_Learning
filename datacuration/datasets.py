#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn import datasets
from sklearn.datasets.samples_generator import make_regression

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


#############################################################################
# Synthetic data
#############################################################################

def regression_data(n_samples, n_features):
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           random_state=0)
    X_sp = sparse.coo_matrix(X)


#############################################################################
# Realworld data
#############################################################################
def diabetes_data():
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = datasets.target


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
