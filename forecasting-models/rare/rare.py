#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from model.model_evaluation import ModelEvaluation
from model.arima import auto_arima

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

"""
RARE options:

> RARE variations:
AR terms:
1. l1 (same for autoregression and exog var)
2. l1 (diff for autoregression and exog var)
3. l2 (same for autoregression and exog var)
4. l2 (diff for autoregression and exog var)
5. l1 and l2 (same for autoregression and exog var)
6. l1 and l2 (diff for autoregression and exog var)

AR + MA terms: 
1. l1 (same for ar, ma and exog var)
2. l1 (diff for ar, ma and exog var)
3. l2 (same for ar, ma and exog var)
4. l2 (diff for ar, ma and exog var)
5. l1 and l2 (same for ar, ma and exog var)
6. l1 and l2 (diff for ar, ma and exog var)

"""


class RARE:
    def __init__(self, ts=None, train_start_date=None, train_end_date=None):
        self.ts = ts
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

    def do_evaluation(self, modeleval: ModelEvaluation, round_value=False,
                      make_nonnegative=False):
        ts_pred = self.fit_and_forecast(modeleval)
        if round_value:
            ts_pred = np.around(ts_pred)
        if make_nonnegative:
            ts_pred[ts_pred < 0] = 0

        # print(ts_pred.head())
        # print(modeleval.ts_test.head())
        modeleval.evaluate(modeleval.ts_test, ts_pred)
        # print("RMSE:", modeleval.rmse)

    def fit_and_forecast(self, modeleval: ModelEvaluation):
        model_param = modeleval.model_param
        test_param = modeleval.test_param
        ts_train = modeleval.ts_train
        ts_test = modeleval.ts_test
        ts_exog = modeleval.ts_exog
        ts_exog_train = modeleval.ts_exog_train
        ts_exog_test = modeleval.ts_exog_test

        print("Length of train =", ts_train.shape[0])
        print("Start/end date =", ts_train.index.min(), ts_train.index.max())
        print("Length of test =", ts_test.shape[0])
        print("Start/end date =", ts_test.index.min(), ts_test.index.max())
        print("Length of train external=", ts_exog_train.shape[0])
        print("Start/end date =", ts_exog_train.index.min(),
              ts_exog_train.index.max())
        print("Length of test external=", ts_exog_test.shape[0])
        print("Start/end date =", ts_exog_test.index.min(),
              ts_exog_test.index.max())

        list_pred = []

        if test_param['sliding_window']:
            print("Sliding window approach")
            if test_param['window_size']:
                window_size = test_param['window_size']
            else:
                window_size = len(ts_train)

            if not test_param["look_ahead_step"]:
                look_ahead_step = 1
            else:
                look_ahead_step = test_param["look_ahead_step"]
            cur_window_train = ts_train[-window_size:]
            # cur_window_data_points = list(ts_train[-window_size:].values)
            # TO DO: need to fix this
            cur_window_train_exog = ts_exog_train[-window_size:]

            for i in range(0, len(ts_test), look_ahead_step):
                y_pred, model = self.fit(cur_window_train,
                                         cur_window_train_exog)
                print(ts_exog_test[i:i + look_ahead_step])

                if i + look_ahead_step > len(ts_test):
                    forecast_res = self.forecast(
                        model, ts_exog_test[i:i + look_ahead_step].values,
                        steps=len(ts_exog_test[i:i + look_ahead_step]))
                else:
                    forecast_res = self.forecast(
                        model, ts_exog_test[i:i + look_ahead_step].values,
                        steps=look_ahead_step)

                list_pred.extend(forecast_res)
                cur_window_train = cur_window_train[
                                   look_ahead_step:]
                temp_val = ts_test[i:i + look_ahead_step]
                cur_window_train = cur_window_train.append(temp_val)

                cur_window_train_exog = cur_window_train_exog[
                                        look_ahead_step:]
                temp_val = ts_exog_test[i:i + look_ahead_step]
                cur_window_train_exog = cur_window_train_exog.append(
                    temp_val)
        else:
            # fit the model
            model = self.fit(ts_train, ts_exog_train,
                             n_lag=model_param['max_p'],
                             algo=model_param['algorithm'])
            print("-" * 75)
            print("Model description:")
            print(model)
            # print("Coefficients:")
            # print(model.coef_)
            ar_coef_ = model.coef_[:model_param['max_p']]
            exog_coef_ = model.coef_[model_param['max_p']:]

            n_sel_var = len(model.coef_[model.coef_ != 0])
            n_sel_ar_var = len(ar_coef_[ar_coef_ != 0])
            n_sel_exog_var = len(exog_coef_[exog_coef_ != 0])
            print("#selected features =", n_sel_var)
            print("#selected ar terms =", n_sel_ar_var)
            print("index:", np.where(ar_coef_[ar_coef_ != 0]))
            print("#selected exog terms =", n_sel_exog_var)
            print("index:", np.where(exog_coef_[exog_coef_ != 0]))

            train_end_date = ts_train.index.max()
            test_end_date = ts_test.index.max()
            gap_and_test_dates = pd.date_range(
                train_end_date + pd.Timedelta(days=1),
                test_end_date)

            gap_start_date = train_end_date + pd.Timedelta(days=1)
            ts_exog_gap_test = ts_exog[gap_start_date:]
            past_y = [x for x in ts_train[-model_param['max_p']:]]
            # past_y = np.array(past_y[::-1])  # order lag 1, lag 2, ...
            past_y = past_y[::-1]  # order lag 1, lag 2, ...

            forecast_res = self.forecast(model, past_y, ts_exog_gap_test,
                                         steps=len(gap_and_test_dates))
            list_pred = forecast_res[-len(ts_test):]

        ts_pred = pd.Series(list_pred[:ts_test.size].values,
                            index=ts_test.index)
        # print(ts_pred)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        return ts_pred

    def fit(self, y, X_exog=None, n_lag=21, algo="lasso-cv", n_cv_fold=10):
        # first n_lag values from y are considered as features
        y_sub = y[n_lag:]
        lagged_y = list()

        for lag in range(1, n_lag + 1):
            temp_y = y.shift(lag)
            temp_y.name = "lag_" + str(lag)
            lagged_y.append(temp_y[n_lag:])
        lagged_y_df = pd.concat(lagged_y, axis=1)

        if X_exog is not None:
            X_exog_sub = X_exog[n_lag:]
            features_df = pd.concat([lagged_y_df, X_exog_sub], axis=1)
        else:
            features_df = lagged_y_df

        model = None
        if algo == "lasso-cv":
            model = self.fit_lasso_cv(features_df.values, y_sub.values,
                                      n_fold=n_cv_fold)
        elif algo == "lasso-lars-cv":
            model = self.fit_lasso_lars_cv(features_df.values, y_sub.values,
                                           n_fold=n_cv_fold)
        elif algo == "lasso-lars-aic":
            model = self.fit_lasso_lars_ic(features_df.values, y_sub.values,
                                           criterion='aic')
        elif algo == "lasso-lars-bic":
            model = self.fit_lasso_lars_ic(features_df.values, y_sub.values,
                                           criterion='bic')
        elif algo == "lars-cv":
            model = self.fit_lars_cv(features_df.values, y_sub.values,
                                     n_fold=n_cv_fold)
        elif algo == "enet-cv":
            model = self.fit_enet_cv(features_df.values, y_sub.values,
                                     n_fold=n_cv_fold)
        elif algo == "ridge-cv":
            model = self.fit_ridge_cv(features_df.values, y_sub.values,
                                      n_fold=n_cv_fold)
        else:
            raise ValueError("Not a valid algorithm.")

        return model

    def forecast(self, model, past_y, X_exog, steps=1):
        print("past y length = ", len(past_y))
        pred_ys = list()
        for exog in X_exog.values:
            x = np.concatenate([np.array(past_y), exog])
            x = x.reshape(1, -1)
            pred_y = model.predict(x)[0]
            pred_ys.append(pred_y)
            past_y.pop()
            past_y.insert(0, pred_y)

        ts_pred = pd.Series(pred_ys, index=X_exog.index)
        return ts_pred

    def fit_lasso_cv(self, X, y, n_fold=10):
        from sklearn.linear_model import LassoCV
        lasso_cv = LassoCV(cv=n_fold)
        lasso_cv.fit(X, y)
        return lasso_cv

    def fit_lasso_lars_cv(self, X, y, n_fold=10):
        from sklearn.linear_model import LassoLarsCV
        lasso_lars_cv = LassoLarsCV(cv=n_fold)
        lasso_lars_cv.fit(X, y)
        return lasso_lars_cv

    def fit_lasso_lars_ic(self, X, y, criterion='aic'):
        from sklearn.linear_model import LassoLarsIC
        lasso_lars_ic = LassoLarsIC(criterion=criterion)
        lasso_lars_ic.fit(X, y)
        return lasso_lars_ic

    def fit_lars_cv(self, X, y, n_fold=10):
        from sklearn.linear_model import LarsCV
        lars_cv = LarsCV(cv=n_fold)
        lars_cv.fit(X, y)
        return lars_cv

    def fit_enet_cv(self, X, y, n_fold=10, l1_ratio=0.5):
        from sklearn.linear_model import ElasticNetCV
        enet_cv = ElasticNetCV(cv=n_fold, l1_ratio=l1_ratio)
        enet_cv.fit(X, y)
        return enet_cv

    def fit_ridge_cv(self, X, y, n_fold=10):
        from sklearn.linear_model import RidgeCV
        ridge_cv = RidgeCV(cv=n_fold)
        ridge_cv.fit(X, y)
        return ridge_cv
