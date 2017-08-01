#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from datacuration.data_struct import TimeSeries

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def load_a_time_series(filepath, header=0, name='count', index_name='date'):
    ts = pd.read_csv(filepath, header=header, parse_dates=True, index_col=0,
                     squeeze=True)
    ts.name = name
    ts.index.name = index_name
    fname = os.path.split(filepath)[1]
    fname_wo_ext = os.path.splitext(fname)[0]
    return TimeSeries(ts, name=fname_wo_ext)


def load_time_series(filelist, header=0, name='count', index_name='date'):
    tslist = list()
    for f in filelist:
        fname = os.path.split(f)[1]
        fname_wo_ext = os.path.splitext(fname)[0]
        ts = pd.read_csv(f, header=header, parse_dates=True, index_col=0,
                         squeeze=True)
        ts.name = name
        ts.index.name = index_name
        cts = TimeSeries(ts, name=fname_wo_ext)
        tslist.append(cts)
    return tslist


def load_ts(filepath, start_date=None, end_date=None):
    print("Loading:", filepath)
    ts = pd.read_csv(filepath, header=0, index_col=0,
                               parse_dates=True, squeeze=True)
    print("Min date =", ts.index.min())
    print("Max date =", ts.index.max())
    print(ts.describe())

    if start_date is not None or end_date is not None:
        ts = ts[start_date:end_date]
    return ts
