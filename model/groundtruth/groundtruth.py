__version__ = 1.0
__author__ = "Tozammel Hossain, Shuyang Gao"
__email__ = "tozammel@isi.edu, sgao@isi.edu"

import pandas as pd
from model.model_evaluation import ModelEvaluation
from datetime import datetime, timedelta


class GroundTruth(object):
    def fit(self, df):
        self.train_start_date = min(df['date'])
        self.train_end_date = max(df['date'])
        self.rate = float(df['count'].sum()) / float(len(df['count']))

    def make_future_dataframe(self, periods=0):
        date = []
        cur_date = self.train_end_date
        for i in range(periods):
            cur_date = cur_date + timedelta(days=1)
            date.append(cur_date)
        return pd.DataFrame({'date': date})

    def predict(self, future):
        count = []
        for i in range(len(future['date'])):
            count.append(self.rate)
        future['count'] = count
        return future

    def do_evaluation(self, model_evaluation, round_value=False,
                      make_nonnegative=False):
        ground_truth = model_evaluation.ts_test
        model_evaluation.evaluate(ground_truth, ground_truth)
