import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score
from pickle import dump


#  --- this model uses the X dataset: all data with no imputing and no synthetic 
X = pd.read_csv('../Resources/X.csv')
y = pd.read_csv('../Resources/y.csv').to_numpy().ravel()

# select just demographic features
Xr = X.iloc[:, np.r_[0:2, 14:30]]

_Xr, Xr_, _y, y_ = tts(Xr, y)
ss = SS()
_Xrs = ss.fit_transform(_Xr)
Xrs_ = ss.transform(Xr_)
rfc = RFC()
rfc.fit(_Xrs, _y)

score = rfc.score(Xrs_, y_)

yp_ = rfc.predict_proba(Xrs_)[:,1]
auc = roc_auc_score(y_, yp_)

if __name__ == '__main__':
    print(score, auc)
    dump(ss, open('../Scalers/d-rfc.pkl', 'wb'))
    dump(rfc, open('../Models/d-rfc.pkl', 'wb'))