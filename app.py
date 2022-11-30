from flask import Flask, request, render_template
from pickle import load

# ss = load(open('Scalers/d-rfc.pkl', 'rb'))
# rfc = load(open('Models/d-rfc.pkl', 'rb'))

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
        entries = {k:v for k,v in request.form.items()}
        features = []
        print(entries)
        prediction_text = ' '.join(f'{k}: {v}' for k, v in entries.items())
        return render_template('predict.html', prediction_text=prediction_text, entries=entries)
    else: 
        return render_template('predict.html', prediction_text='Make a prediction!')

# --- following '/models/' routes and for each model - all static
# knn
@app.route('/models/knn')
def knn():
    return render_template('model/knn.html')

# logistic regression
@app.route('/models/logistic-regression')
def logistic_regression():
    return render_template('model/logistic_regression.html')

# nueral net
@app.route('/models/neural_network')
def neural_network():
    return render_template('model/neural_network.html')

# random forest classifier
@app.route('/models/random-forest-classifier')
def random_forest_classifier():
    return render_template('model/random_forest.html')

if __name__ == '__main__':
    app.run(debug=True)