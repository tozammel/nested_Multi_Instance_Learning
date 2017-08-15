#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

import numpy as np
import pandas as pd
from model.model_evaluation import ModelEvaluation
from statsmodels.nonparametric.kde import KDEUnivariate


class WeeklyKDE:
    def __init__(self):
        pass

    def do_evaluation(self, modeleval: ModelEvaluation, round_value=False,
                      make_nonnegative=False):
        # print("Fitting:", self.__class__)
        ts_pred = self.fit(modeleval)
        if round_value:
            ts_pred = np.around(ts_pred)
        if make_nonnegative:
            ts_pred[ts_pred < 0] = 0
        modeleval.evaluate(modeleval.ts_test, ts_pred)

    def fit(self, modeleval: ModelEvaluation):
        model_param = modeleval.model_param
        ts_train = modeleval.ts_train
        ts_test = modeleval.ts_test

        # create daywise data
        ts_dayofweek = ts_train.groupby(ts_train.index.dayofweek)
        train_end_date = ts_train.index.max()
        test_end_date = ts_test.index.max()

        gap_and_test_dates = pd.date_range(
            train_end_date + pd.Timedelta(days=1),
            test_end_date)
        gap_and_test_dates_dayofweek = gap_and_test_dates.groupby(
            gap_and_test_dates.dayofweek
        )

        test_dates_dayofweek = ts_test.index.groupby(ts_test.index.dayofweek)
        ts_pred = pd.Series()

        for day, ts_day in ts_dayofweek:
            freq = ts_day.value_counts()
            # reindex the freq
            freq = freq.reindex(
                np.arange(freq.index.min(), freq.index.max() + 1),
                fill_value=0)
            freq_dist = freq / freq.sum()
            # pdf = KDEUnivariate(ts_day.astype(float))
            # pdf.fit()

            indx_gap_test_dayofweek = gap_and_test_dates_dayofweek[day]
            indx_test_dayofweek = test_dates_dayofweek[day]

            # forecast_res = np.random.choice(freq_dist.index.values,
            #                                 size=len(indx_gap_test_dayofweek),
            #                                 p=freq_dist.values)
            """
            Rather than taking a sample and take a set of sample and accepth 
            the mode as the value
            """
            from scipy.stats import mode
            forecast_res = list()
            for aday in range(len(indx_gap_test_dayofweek)):
                fcast = np.random.choice(freq_dist.index.values, size=5,
                                         p=freq_dist.values)
                print(fcast)
                mode_val = mode(fcast)[0][0]
                forecast_res.append(mode_val)

            list_pred = forecast_res[-len(indx_test_dayofweek):]
            ts_pred_sub = pd.Series(list_pred, index=indx_test_dayofweek)
            ts_pred = ts_pred.append(ts_pred_sub)

        if "pred_days" in model_param:
            pass

        ts_pred = ts_pred.sort_index()
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        # print(ts_pred)
        # input("KDE method")
        return ts_pred
