from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load('model.joblib')
encoders = joblib.load('encoders.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()

    data = []
    for col in ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']:
        value = input_data[col]
        encoded = encoders[col].transform([value])[0]
        data.append(encoded)

    final_input = np.array([data])
    prediction = model.predict(final_input)

    result = encoders['class'].inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)