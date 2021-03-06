#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, LassoLars, \
    LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.linear_model import ElasticNet

from datacuration import synthetic_data as syn_data

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

"""
http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html
http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html

"
Use the Akaike information criterion (AIC), the Bayes Information criterion 
(BIC) and cross-validation to select an optimal value of the regularization 
parameter alpha of the Lasso estimator.

Information-criterion based model selection:
 * very fast 
 * relies on a proper estimation of degrees of freedom 
 * derived for large samples (asymptotic results) and assume the model is 
   correct, i.e. that the data are actually generated by this model. 
 * Tend to break when the problem is badly conditioned (more features than 
   samples).

Lasso: select #features max of sample size

Options:

LassoLarsIC: 
* based on AIC/BIC criteria

LassoLarsCV:
* based on cross-validation
* least angle regression

LassoCV: 
* based on cross-validation
* coordinate descent

"""


def run_lasso(X_train, y_train, X_test, y_test):
    alpha = 0.1
    lasso = Lasso(alpha=alpha)
    model = lasso.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_score_lasso = r2_score(y_test, y_pred)
    print(lasso)
    print("r^2 on test data : %f" % r2_score_lasso)
    print(r2_score_lasso)


def run_lasso_cv(X_train, y_train, X_test, y_test):
    """
    use coordinate descent
    :param X_train: 
    :param y_train: 
    :param X_test: 
    :param y_test: 
    :return: 
    """
    model = LassoCV(cv=10).fit(X_train, y_train)
    print(model.alpha_)
    print(model.alphas_)


def run_lasso_lars_cv(X_train, y_train, X_test, y_test):
    """
    :param X_train: 
    :param y_train: 
    :param X_test: 
    :param y_test: 
    :return: 
    """
    model_lars_cv = LassoLarsCV(cv=10)
    model_lars_cv.fit(X_train, y_train)
    print(model_lars_cv.alpha_)
    print(model_lars_cv.cv_alphas_)
    print(model_lars_cv.cv_mse_path_)


def run_lasso_lars_ic(X_train, y_train, X_test, y_test):
    """
    ic: information criterion (AIC/BIC)
    usually faster, but breaks up when sample size << feature size
    :param X_train: 
    :param y_train: 
    :param X_test: 
    :param y_test: 
    :return: 
    """
    model_bic = LassoLarsIC(criterion="bic")
    model_aic = LassoLarsIC(criterion="aic")

    model_bic.fit(X_train, y_train)
    model_aic.fit(X_train, y_train)

    plot_ic_criterion(model_bic, 'BIC', 'b')
    plot_ic_criterion(model_aic, 'AIC', 'r')


def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')


def run_enet(X_train, y_train, X_test, y_test):
    alpha = 0.1
    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
    model = enet.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_score_enet = r2_score(y_test, y_pred)
    print(enet)
    print("r^2 on test data : %f" % r2_score_enet)


def simulated_data():
    n_samples, n_features = 50, 200
    X, y = syn_data.sim_reg_sparse_data(n_samples=n_samples,
                                        n_features=n_features, seed=42,
                                        n_sparse_features=10)

    X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
    X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

    return X_train, y_train, X_test, y_test


def main(args):
    X_train, y_train, X_test, y_test = simulated_data()
    run_lasso(X_train, y_train, X_test, y_test)
    run_enet(X_train, y_train, X_test, y_test)
    run_lasso_cv(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
