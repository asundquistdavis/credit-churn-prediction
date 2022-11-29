from flask import Flask, request, render_template
from pickle import load

ss = load(open('Scalers/d-rfc.pkl', 'rb'))
rfc = load(open('Models/d-rfc.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = {k:v for k,v in zip(request.form.keys(), request.form.values())}
    features = []
    prediction_text = " ".join([feature for feature in features])
    return render_template('index.html', prediction_text=prediction_text, features=features)

if __name__ == '__main__':
    app.run(debug=True)