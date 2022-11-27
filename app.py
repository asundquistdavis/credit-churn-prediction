from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    prediction_text = " ".join([feature for feature in features])
    return render_template('index.html', prediction_text=prediction_text, features=features)

if __name__ == '__main__':
    app.run(debug=True)