#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

import numpy as np
import pandas as pd
from model.model_evaluation import ModelEvaluation


class Average:
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

        if 'window_size' in model_param:
            window_size = model_param['window_size']
        else:
            window_size = len(ts_train)

        if 'sliding_window' in model_param:
            sw = model_param['sliding_window']
            leap = model_param['leap']
            if sw:
                cur_window_data_points = ts_train[-window_size:].values
                cur_rolling_val = np.mean(cur_window_data_points)
                train_point = list(range(ts_test.size))
                list_pred = []
                for i in range(ts_test.size):
                    list_pred.append(cur_rolling_val)

        if 'pred_days' in model_param:
            pred_days = model_param['pred_days']
            rolling_values = []
            for i in range(window_size):
                rolling_values.append(ts_train[-window_size + i])
            # rolling_values = ts_train[-window_size:]
            Npreds = 0
            avg_val = float(sum(rolling_values)) / len(rolling_values)
            list_pred = []
            for i in range(ts_test.size):
                list_pred.append(avg_val)
                Npreds += 1
                if Npreds == pred_days:
                    for j in range(pred_days):
                        if rolling_values != []:
                            rolling_values.pop(0)
                    if pred_days < window_size:
                        for j in range(pred_days):
                            rolling_values.append(
                                ts_test[i - pred_days + j + 1])
                    else:
                        for j in range(window_size):
                            rolling_values.append(
                                ts_test[i - window_size + j + 1])
                    avg_val = float(sum(rolling_values)) / float(
                        len(rolling_values))
                    Npreds = 0

        else:
            ts = ts_train[-window_size:]
            avg_val = ts.mean()
            list_pred = [avg_val] * ts_test.size
        ts_pred = pd.Series(list_pred, index=ts_test.index)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        return ts_pred


class WeeklyAverage:
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

        if 'window_size' in model_param:
            window_size = model_param['window_size']
        else:
            window_size = len(ts_train)

        if 'sliding_window' in model_param:
            sw = model_param['sliding_window']
            leap = model_param['leap']
            if sw:
                cur_window_data_points = ts_train[-window_size:].values
                cur_rolling_val = np.mean(cur_window_data_points)
                train_point = list(range(ts_test.size))
                list_pred = []
                for i in range(ts_test.size):
                    list_pred.append(cur_rolling_val)

        if 'pred_days' in model_param:
            pred_days = model_param['pred_days']
            rolling_values = []
            for i in range(window_size):
                rolling_values.append(ts_train[-window_size + i])
            # rolling_values = ts_train[-window_size:]
            Npreds = 0
            avg_val = float(sum(rolling_values)) / len(rolling_values)
            list_pred = []
            for i in range(ts_test.size):
                list_pred.append(avg_val)
                Npreds += 1
                if Npreds == pred_days:
                    for j in range(pred_days):
                        if rolling_values != []:
                            rolling_values.pop(0)
                    if pred_days < window_size:
                        for j in range(pred_days):
                            rolling_values.append(
                                ts_test[i - pred_days + j + 1])
                    else:
                        for j in range(window_size):
                            rolling_values.append(
                                ts_test[i - window_size + j + 1])
                    avg_val = float(sum(rolling_values)) / float(
                        len(rolling_values))
                    Npreds = 0

        else:
            ts = ts_train[-window_size:]

            df_dayofweek = ts.groupby(
                ts.index.dayofweek).agg(
                ['count', 'sum', 'mean', 'std', 'min', 'max'])
            df_dayofweek.columns = ['num_year_weekday', 'sum', 'mean', 'std',
                                    'min', 'max']
            df_dayofweek.index.name = 'weekday'

            # avg_val = ts.mean()
            list_pred = list()
            for adate in ts_test.index:
                day = adate.dayofweek
                list_pred.append(df_dayofweek.ix[day, 'mean'])
        ts_pred = pd.Series(list_pred, index=ts_test.index)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        return ts_pred
