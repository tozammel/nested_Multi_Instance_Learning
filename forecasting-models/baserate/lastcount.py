#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__version__ = 1.0
__author__ = "Tozammel Hossain, Shuyang Gao"
__email__ = "tozammel@isi.edu, sgao@isi.edu"

import pandas as pd
from model.model_evaluation import ModelEvaluation


class LastCount:
    def __init__(self):
        pass

    def do_evaluation(self, modeleval: ModelEvaluation):
        ts_pred = self.fit(modeleval)
        modeleval.evaluate(modeleval.ts_test, ts_pred)

    def fit(self, modeleval: ModelEvaluation):
        model_param = modeleval.model_param
        ts_train = modeleval.ts_train
        ts_test = modeleval.ts_test
        if "pred_days" in model_param:
            pred_days = model_param["pred_days"];
            last_count = ts_train[-1];
            Npreds = 0
            list_pred = [];
            for i in range(ts_test.size):
                list_pred.append(last_count);
                Npreds += 1;
                if Npreds == pred_days:
                    last_count = ts_test[i];
                    Npreds = 0;
        else:
            last_count = ts_train[-1]
            list_pred = [last_count] * ts_test.size
        ts_pred = pd.Series(list_pred, index=ts_test.index)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        return ts_pred
