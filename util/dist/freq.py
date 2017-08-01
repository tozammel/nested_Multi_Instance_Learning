#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
import pandas as pd

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def freq_dist_by_range(x, bins=None):
    if bins is None:
        hist = np.histogram(x)
    else:
        hist = np.histogram(x, bins=bins)

    bins = hist[1]
    freq = hist[0]
    d = OrderedDict()
    for i, val in enumerate(bins[:-1]):
        key = str(val) + "-" + str(bins[i + 1] - 1)
        d[key] = freq[i]
    df = pd.DataFrame.from_dict(d, orient='index')
    df.columns = ['count']
    df.index.name = 'range'
    return hist, df
