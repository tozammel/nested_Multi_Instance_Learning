#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class MetaDatasetClass:
    def __init__(self, filepath, name, country=None, city=None, actor=None,
                 exogfilepath=None):
        self.filepath = filepath
        self.name = name
        self.country = country
        self.city = city
        self.actor = actor
        self.exog_filepath = exogfilepath


class CustomTimeSeries:
    def __init__(self, ts=None, name=None, type=None):
        """

        :type type: str
        """
        self.ts = ts
        self.name = name
        self.type = type

    def set_ts(self, ts):
        self.ts = ts


class TimeSeries:
    def __init__(self, ts, name=None, start_date=None, end_date=None):
        self.ts = ts
        self.name = name
        self.start_date = start_date
        self.end_date = end_date

    def plot(self, outfilepath=None):
        pass

    def describe(self):
        print("Name =", self.name)
        print("#entries =", self.ts.size)
        print("first date =", self.ts.index.min())
        print("last date =", self.ts.index.max())
        print("start date =", self.start_date)
        print("end date =", self.end_date)

    def set_start_date(self, start_date):
        self.start_date = start_date

    def set_end_date(self, end_date):
        self.end_date = end_date

    def set_start_end_date(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def get_sliced_ts(self):
        if self.start_date is None:
            sd = self.ts.index.min()
        else:
            sd = self.start_date

        if self.end_date is None:
            ed = self.ts.index.max()
        else:
            ed = self.end_date
        return self.ts[sd: ed]
