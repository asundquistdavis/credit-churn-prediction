import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score
from pickle import dump

#  --- this model uses the X dataset (drop target, id and Niave Bayes columns with 1-hot rencode categorical values, no imputing and no synthetic samples) 
X = pd.read_csv('../Resources/X.csv')
y = pd.read_csv('../Resources/y.csv').to_numpy().ravel()

# select just demographic features
Xr = X.iloc[:, np.r_[0:2, 14:33]]

# save the name of the features
demographics = Xr.columns

# split data into training and testing
_Xr, Xr_, _y, y_ = tts(Xr, y)

# scale data
ss = SS()
_Xrs = ss.fit_transform(_Xr)
Xrs_ = ss.transform(Xr_)

# create rfc and fit with training data
rfc = RFC()
rfc.fit(_Xrs, _y)

# get acccuracy score of the model
score = rfc.score(Xrs_, y_)

# get auc-roc score of the model
yp_ = rfc.predict_proba(Xrs_)[:,1]
auc = roc_auc_score(y_, yp_)

# make a test prediction
test_a = rfc.predict(np.array([[27, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]))

# save the test in data
D_ = pd.DataFrame(ss.inverse_transform(Xrs_), columns=demographics)
D_['target'] = y_
D_['prediction'] = yp_

# run the model
if __name__ == '__main__':
    
    # print scores
    print(score, auc)

    # print test prediction 
    print(test_a)

    # save the model and scaler as pickle files and the test data as csv
    dump(ss, open('../Scalers/d-rfc.pkl', 'wb'))
    dump(rfc, open('../Models/d-rfc.pkl', 'wb'))
    D_.to_csv('../Resources/D_.csv')