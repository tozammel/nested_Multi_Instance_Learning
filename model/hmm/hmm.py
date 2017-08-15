#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__version__ = 1.0
__author__ = "Tozammel Hossain, Shuyang Gao"
__email__ = "tozammel@isi.edu, sgao@isi.edu"

from scipy.optimize import minimize
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from model.hmm.hmm_models import GaussianHMM, PoissonHMM, \
    GeometricHMM, HurdleGeometricHMM
from model.model_evaluation import ModelEvaluation
from model.arguments import ModelArguments
from model.model_results import ModelResults


class HMM:
    @staticmethod
    def sample_dates(dates, start_date, end_date, emission='Gaussian',
                     algorithm='viterbi'):
        models = {'Gaussian': GaussianHMM, 'Poisson': PoissonHMM,
                  'Geometric': GeometricHMM,
                  'HurdleGeometric': HurdleGeometricHMM}
        hmm_model = models[emission]
        print("Total event counts:", len(dates))
        dates_conv = []
        for i in dates:
            dates_conv.append(datetime.strptime(i, "%Y-%m-%d"))
        dates_conv.sort()
        date_idx = []
        for i in range(len(dates_conv)):
            date_idx.append((dates_conv[i] - dates_conv[0]).days)

        # preparing data for HMM
        data = [1]
        for i in range(1, len(date_idx)):
            if date_idx[i] != date_idx[i - 1]:
                for j in range(date_idx[i - 1] + 1, date_idx[i]):
                    data.append(0)
                data.append(1)
            else:
                data[-1] += 1
        data = [[i] for i in data]
        if len(data) == 1:
            data = [[0], data[0]]

        # learning HMM
        print("start learning parameters for HMM model...")
        model = hmm_model(n_components=2, algorithm=algorithm).fit(data)
        if hmm_model == GaussianHMM:
            e0 = model.means_[0][0]
            e1 = model.means_[1][0]
        elif hmm_model == PoissonHMM:
            e0 = model.means_[0][0]
            e1 = model.means_[1][0]
        elif hmm_model == GeometricHMM:
            e0 = 1.0 / (1.0 - model.means_[0][0]) - 1.0
            e1 = 1.0 / (1.0 - model.means_[1][0]) - 1.0
        elif hmm_model == HurdleGeometricHMM:
            e0 = 1.0 / (1.0 - model.mu_[0][0]) * model.gamma_[0][0]
            e1 = 1.0 / (1.0 - model.mu_[1][0]) * model.gamma_[1][0]
        print("finish learning parameters...")
        print("Expected events per day in ACTIVE state: ", max(e0, e1))
        print("Expected events per day in INACTIVE state: ", min(e0, e1))

        # decoding HMM
        states = model.predict(data)

        # predicting HMM
        p0 = model.transmat_[states[-1], 0]  # probability in state 0
        p1 = model.transmat_[states[-1], 1]  # probability in state 1
        new_dates = []
        for i in range(date_idx[-1] + 1,
                       (datetime.strptime(end_date, '%Y-%m-%d') -
                            dates_conv[0]).days + 1):
            n_pred_events = p0 * e0 + p1 * e1
            if i >= (
                        datetime.strptime(start_date, '%Y-%m-%d') - dates_conv[
                        0]).days:
                for j in range(int(n_pred_events)):
                    new_dates.append(dates_conv[0] + timedelta(days=i))
                    new_dates[-1] = str(new_dates[-1].date())
            pp0 = p0 * model.transmat_[0, 0] + p1 * model.transmat_[1, 0]
            pp1 = p0 * model.transmat_[0, 1] + p1 * model.transmat_[1, 1]
            p0 = pp0
            p1 = pp1

        return np.array(new_dates)

    def __init__(self):
        pass

    def do_evaluation(self, modeleval: ModelEvaluation, round_value=False,
                      make_nonnegative=False):
        ts_pred = self.fit_and_forecast(modeleval)
        if round_value:
            ts_pred = np.around(ts_pred)
        if make_nonnegative:
            ts_pred[ts_pred < 0] = 0
        modeleval.evaluate(modeleval.ts_test, ts_pred)

    # def fit(self, model_param, ts_train, ts_test):
    def fit_and_forecast(self, modeleval: ModelEvaluation):
        model_param = modeleval.model_param
        test_param = modeleval.test_param
        ts_train = modeleval.ts_train
        ts_test = modeleval.ts_test

        if test_param["verbose"]:
            print("Fitting with HMM")

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

            cur_window_data_points = [[x] for x in ts_train]

            for i in range(0, len(ts_test), look_ahead_step):
                # train
                hmm_model, model, e = self.learn_param(cur_window_data_points,
                                                       model_param['num_states'],
                                                       model_param['emission'])
                if test_param["verbose"]:
                    print("Expected num of events =", e)

                # predict
                history = cur_window_data_points.copy()
                for j in range(look_ahead_step):
                    states = model.predict(history)

                    last_day_state = states[-1]
                    idx = np.argsort(e)

                    n_pred_events = 0
                    for k in range(model_param['num_states']):
                        p = model.transmat_[last_day_state, k]  # P(last state = i)
                        n_pred_events += p * e[k]

                    list_pred.append(n_pred_events)
                    history.append([n_pred_events])

                    cur_window_data_points.pop(0)

                # update training set
                temp_val = [[x] for x in ts_test[i:i+look_ahead_step].values]
                cur_window_data_points.extend(temp_val)

        else:

            history = [[x] for x in ts_train]
            hmm_model, model, e = self.learn_param(history,
                                                   model_param['num_states'],
                                                   model_param['emission'])

            train_end_date = ts_train.index.max()
            test_start_date = ts_test.index.min()
            # print(train_end_date)
            # print(test_start_date)

            gap_dates = pd.date_range(train_end_date + pd.Timedelta(days=1),
                                      test_start_date, closed="left")
            # predict events for the gaps
            for adate in gap_dates:
                states = model.predict(history)
                last_day_state = states[-1]
                idx = np.argsort(e)
                # # states in the order of 0, 1, 2, ...
                true_last_state = np.where(idx == last_day_state)[0][0]

                n_pred_events = 0
                for i in range(model_param['num_states']):
                    p = model.transmat_[last_day_state, i]  # P(last state = i)
                    n_pred_events += p * e[i]
                history.append([n_pred_events])

            if "pred_days" in model_param:
                previous_days = len(history)
                for i in range(ts_test.size):
                    history.append([ts_test[i]])

                # predict_new([a1, a2, a3, a4, a5, ...]) = [predict([a1])[-1], predict([a1,a2])[-1], predict([a1,a2,a3])[-1],predict([a1,a2,a3,a4])[-1],...]
                states = model.predict_new(history)
                predictions = []
                pred_days = model_param['pred_days']
                Npreds = 0
                last_day_idx = previous_days - 1
                while last_day_idx < ts_test.size + previous_days:
                    last_day_state = states[last_day_idx]
                    p = []
                    for i in range(model_param['num_states']):
                        p.append(model.transmat_[last_day_state, i])
                    for i in range(pred_days):
                        n_pred_events = 0
                        for j in range(model_param['num_states']):
                            n_pred_events += p[j] * e[j]
                        if last_day_idx + i + 1 < ts_test.size + previous_days:
                            predictions.append(n_pred_events)

                        # update state probability
                        pp = []
                        for j in range(model_param['num_states']):
                            pp.append(0.0)
                            for k in range(model_param['num_states']):
                                pp[j] += p[k] * model.transmat_[k, j]
                        for j in range(model_param['num_states']):
                            p[j] = pp[j]
                    last_day_idx += pred_days
                df_pred = pd.DataFrame({'count': predictions}, index=ts_test.index)
                # df_pred.set_index(ts_test.index)
            else:
                # predictions = []

                if ("verbose" in model_param) and (model_param["verbose"]):
                    print(len(ts_test.index))

                for adate in ts_test.index:
                    states = model.predict(history)
                    last_day_state = states[-1]
                    idx = np.argsort(e)
                    # # states in the order of 0, 1, 2, ...
                    true_last_state = np.where(idx == last_day_state)[0][0]

                    n_pred_events = 0
                    for i in range(model_param['num_states']):
                        p = model.transmat_[last_day_state, i]  # P(last state = i)
                        n_pred_events += p * e[i]
                    history.append([n_pred_events])
                    # predictions.append((adate.date(), n_pred_events))
                    list_pred.append(n_pred_events)

                    if ("verbose" in model_param) and (model_param["verbose"]):
                        print(adate.date(),
                              'predicted=%f, expected=%f' % (
                                  n_pred_events, ts_test.ix[adate]))

                # filename = ts_name + "_pred_hmm_v2_w_" + str(size) + \
                #            "_z_" + str(num_states) + "_e_" + emission + ".csv"
                # return pd.DataFrame(predictions), filename
                # df_pred = pd.DataFrame(predictions, columns=['date', 'count'])
                # df_pred.set_index('date', inplace=True)
        ts_pred = pd.Series(list_pred[:ts_test.size], index=ts_test.index)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        # return df_pred['count']
        return ts_pred

    def learn_param(self, hmm_data, num_states, emission, n_iter=50):
        models = {'Gaussian': GaussianHMM, 'Poisson': PoissonHMM,
                  'Geometric': GeometricHMM,
                  'HurdleGeometric': HurdleGeometricHMM}

        hmm_model = models[emission]
        model = hmm_model(n_components=num_states,
                          algorithm='viterbi',
                          n_iter=50).fit(hmm_data)

        e = []  # expected number of events

        for i in range(num_states):
            if hmm_model == GaussianHMM or hmm_model == PoissonHMM:
                e.append(model.means_[i][0])
            elif hmm_model == GeometricHMM:
                e.append(1.0 / (1.0 - model.means_[i][0]) - 1.0)
            elif hmm_model == HurdleGeometricHMM:
                e.append(1.0 / (1.0 - model.mu_[i][0]) * model.gamma_[i][0])

        # states = model.predict(hmm_data)
        # last_day_state = states[-1]
        # idx = np.argsort(e)
        # # states in the order of 0, 1, 2, ...
        # true_last_state = np.where(idx == last_day_state)[0][0]
        #
        # n_pred_events = 0
        # for i in range(num_states):
        #     p = model.transmat_[last_day_state, i]  # P(last state = i)
        #     n_pred_events += p * e[i]

        # return hmm_model, model, e, true_last_state, n_pred_events
        return hmm_model, model, e

    def fit(self, modelarg: ModelArguments):
        model_param = modelarg.model_param
        ts_train = modelarg.endog[
                   modelarg.train_start_date:modelarg.train_end_date]
        if 'window_size' in model_param:
            window_size = model_param['window_size']
        else:
            window_size = len(ts_train)
        cur_window_data_points = ts_train[-window_size:].values

        history = [[x] for x in ts_train]
        hmm_model, model, e = self.learn_param(history,
                                               model_param['num_states'],
                                               model_param['emission'])

        est_param = {"window_size": window_size,
                     "num_states": model_param['num_states'],
                     "emission": model_param['emission'],
                     "model": model,
                     "exp_count": e
                     }

        modelres = ModelResults(model_name="HMM",
                                estimated_param=est_param)
        return modelres

    def forecast(self, modelarg: ModelArguments, modelres: ModelResults):
        gap_dates = pd.date_range(
            modelarg.train_end_date + pd.Timedelta(days=1),
            modelarg.test_start_date, closed='left')
        concerned_dates = pd.date_range(modelarg.test_start_date,
                                        modelarg.test_end_date)
        gap_and_concerned_dates = gap_dates.append(concerned_dates)
        est_param = modelres.learned_param






        list_pred = [est_param['rate']] * gap_and_concerned_dates.size
        ts_pred = pd.Series(list_pred[-concerned_dates.size:],
                            index=concerned_dates)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        return ts_pred
