# imports
import csv
from sklearn.metrics import confusion_matrix as cm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS

from sklearn.linear_model import LogisticRegression as LRC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression as MLR
model_dict = {'log': LRC, 'rfc': RFC, 'knn': KNC, 'svc': SVC, 'mlr': MLR}

import math
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV as RSCV

# combines train-test-split and standard scaler
def split_and_scale(X, y):
    _X, X_, _y, y_ = tts(X, y)
    ss = SS()
    _Xs = ss.fit_transform(_X)
    Xs_ = ss.transform(X_)
    return _Xs, Xs_, _y, y_

# returns length N series (generator) of thresholds in increments of 1/N
def Hs(N):
    return (i/N for i in range(N+1))

# computes 'integral' of 'Y of X' using trapazoid rule for approximation
def auc(X, Y):
#   each 'dx' in 'dX' is the change in x value for each interval
    dX = [b - a for a, b in zip([0] + X, X + [0])][1:-1]
#   each 'foy' in 'foY' is the integrand - f(a) + f(b) /2 - for each interval
    foY = [(yoa + yob)/2 for yoa, yob in zip([0] + Y, Y + [0])][1:-1]
#   the return is the sum of the areas of each trapazoid, absolute value is taken as the direction of integration is not know
    return abs(sum(foy * dx for foy, dx in zip(foY, dX)))
    
# returns the list of 
def roc(X_, y_, model_inst, N, plot = False, area=False, save_path=None):
#   get thresholds
    H = Hs(N)
#   initialize lists
    F = []
    T = []
#   get probabilities of positives
    y_p = model_inst.predict_proba(X_)[:,1]
#   loop through all thresholds
    for t in H:
#       generate false positive and true positve rates: fpr, tpr
        tn, fp, fn, tp = cm(y_, list(map(lambda p: 1 if p >= t else 0, y_p))).ravel()
        fpr = fp/(tn+fp)
        tpr = tp/(tp+fn)
#       append lists
        F.append(fpr)
        T.append(tpr)
#   plot if necessary
    if plot == True:
        plt.plot(F, T)
        plt.xlabel('False Positve Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC for {str(model)[:-2]} Model')
        plt.show()
#   save to path name 'save_path' if provided
    if not save_path == None:
        data = pd.DataFrame({'fpr': F, 'tpr': T})
        data.to_csv(save_path, index=False)
#   return area under curve if auc is selected
    if area:
        return auc(F, T)
#   otherwise return the fpr and tpr values
    else:
        return (F, T)

# returns accuracy from a data set and type of model
def acc_test(X, y, model='log', inst_num=None, **kwargs):
    y = y.to_numpy().ravel()
    _Xs, Xs_, _y, y_ = split_and_scale(X, y)
    M = model_dict[model]
    m = M(**kwargs)
    m.fit(_Xs, _y)
    return m.score(Xs_, y_)
    
# returns an AUC-ROC score from a data set and type of model
def auc_test(X, y, data_set=False, model='log', inst_num=None, trials=False, **kwargs):
    y = y.to_numpy().ravel()
    _Xs, Xs_, _y, y_ = split_and_scale(X, y)
    M = model_dict[model]
    m = M(**kwargs)
    m.fit(_Xs, _y)
    if trials:
        return roc(Xs_, y_, m, 100, area=True)
    save_path = f'../Outputs/{data_set+"_" if data_set else""}{str(m).split("(")[0]}_ROC'
    if not inst_num == None:
        save_path += f'_{inst_num}'
    return roc(Xs_, y_, m, 100, save_path=save_path+'.csv', area=True)

# returns feature importances of a data set trained on a type of model
def feature_importances(X, y, model='log', **kwargs):
    y = y.to_numpy().ravel()
    _Xs, Xs_, _y, y_ = split_and_scale(X, y)
    M = model_dict[model]
    m = M(**kwargs)
    m.fit(_Xs, _y)
    return m.feature_importances_

# scrapes the kaggle page that host the data set for the data dictionary provided there
def get_data_dict(save_path=None):
    import pandas as pd
    from splinter import Browser
    from webdriver_manager.chrome import ChromeDriverManager
    executable_path = {'executable_path': ChromeDriverManager().install()}
    with Browser('chrome', **executable_path, headless=False) as browser:
        browser.visit('http://www.kaggle.com/datasets/whenamancodes/credit-card-customers-prediction')
        data_dict = pd.read_html(browser.html)[0]
    pd.set_option('display.max_colwidth', None)
    if not save_path == None:
        data_dict.to_csv(save_path)
    return data_dict

# does a cross validation search for 'best' hyper parameters
# this function was not used in final version of the project but could be interesting part of further research
def rfc_cv(_X, X_, _y, y_, data_set_name):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rfc = RFC()
    rfc_search = RSCV(estimator = rfc,
                     param_distributions = random_grid,
                     n_iter = 25, 
                     cv = 5,
                     verbose=3,  
                     n_jobs = -1)
    rfc_search.fit(_X, _y)
    best_params = rfc_search.best_params_
    rfc_best = rfc_search.best_estimator_
    score = rfc_best.score(X_, y_)
    auc_ = roc(X_, y_, rfc_best, 100, area=True, save_path=f'../Resources/{data_set_name}_RFC_CV')
    return {'best_estimator': rfc_best, 'best_params': best_params, 'score': score, 'auc': auc_}