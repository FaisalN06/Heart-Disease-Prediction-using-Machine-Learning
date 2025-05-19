import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('RF.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/menu')
def menu():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if output == 0:
        text = "you don't have any symptoms of a Heart Disease."
    else:
        text = "you have the symptoms of having Heart Disease. Please consult a Doctor."
    return render_template('result.html', prediction_text='The model predicted {}'.format(text))

if __name__ == "__main__":
    app.run(debug=True)
