from flask import Flask, request, render_template
import numpy as np
from pickle import load

ss = load(open('Scalers/d-rfc.pkl', 'rb'))
rfc = load(open('Models/d-rfc.pkl', 'rb'))

# this casts text entires as ints
def is_valid(entry):
    if not entry['age'].isnumeric():
        return False, 'Please enter a number for age.'
    elif not entry['numdep'].isnumeric():
        return False, 'Please enter a number for number of dependents.'
    else:
        entry['age'] = int(entry['age'])
        entry['numdep'] = int(entry['numdep'])
        return True, ''

# this converts entries <dict> into features <list/1d array>
def features_from(entry):
    features = []
    features.append(entry['age'])
    features.append(entry['numdep'])
    gender_map = {'m': [0, 1], 'f': [1, 0]}    
    edclvl_map = {'u': [0, 0, 0, 0, 0, 1, 0], 'c': [1, 0, 0, 0, 0, 0, 0], 'd': [0, 1, 0, 0, 0, 0, 0], 'g': [0, 0, 1, 0, 0, 0, 0], 'h': [0, 0, 0, 1, 0, 0, 0], 'p': [0, 0, 0, 0, 1, 0, 0]}
    marsta_map = {'d': [1, 0, 0, 0], 'm': [0, 1, 0, 0], 's': [0, 0, 1, 0]}
    income_map = {'i1': [0, 0, 0, 0, 1, 0], 'i2': [0, 1, 0, 0, 0, 0], 'i3': [0, 0, 1, 0, 0, 0], 'i4': [0, 0, 0, 1, 0, 0], 'i5': [1, 0, 0, 0, 0, 0]}
    features += gender_map[entry['gender']]
    features += edclvl_map[entry['edclvl']]
    features += marsta_map[entry['marsta']]
    features += income_map[entry['income']]
    return np.array([features])

MODEL_TRHESHOLD = .9

app = Flask(__name__)

# home page with infor about project - static
@app.route('/')
def about():
    return render_template('about.html')

# proposed about the data
# TK

# interactive prediction page - dynamic
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # get user entry from form 
        entry = {k:v for k,v in request.form.items()}

        # check if age and number of deps. are numeric
        if not is_valid(entry)[0]:
            return render_template('predict.html', prediction_text=is_valid(entry)[1], entry=entry)
        
        # build feature array from entry
        features = features_from(entry)

        # scale features using the model's scaler
        features_scaled = ss.transform(features)

        print(rfc.predict(features_scaled)[0])

        # use the model to make a prediction based on the users entry/features
        outcomes = ['high risk attrition customer', 'low risk attrition customer']
        prediction = outcomes[(0 if rfc.predict_proba(features_scaled)[0,1] < MODEL_TRHESHOLD else 1)]

        prediction_text = f'{prediction.capitalize()}.'
        return render_template('predict.html', prediction_text=prediction_text, entry=entry)
    else: 
        return render_template('predict.html', prediction_text='Make a prediction!')

# --- following '/models/' routes and for each model - all static
# knn
@app.route('/model/knn')
def knn():
    return render_template('model/knn.html')

# logistic regression
@app.route('/model/logistic-regression')
def logistic_regression():
    return render_template('model/logistic_regression.html')

# nueral net
@app.route('/model/neural-network')
def neural_network():
    return render_template('model/neural_network.html')

# random forest classifier
@app.route('/model/random-forest-classifier')
def random_forest_classifier():
    return render_template('model/random_forest.html')

# icon logo
@app.route('..Resources/churn.ico')
def icon():
    return render_template('..Resources/churn.ico')


if __name__ == '__main__':
    app.run(debug=True)