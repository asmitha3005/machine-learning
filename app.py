from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load('model.joblib')

mapping = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()

        data = []
        for col in mapping:
            data.append(mapping[col][input_data[col]])

        final_input = np.array([data])
        prediction = model.predict(final_input)

        return render_template('index.html', prediction_text=f"Prediction: {prediction[0]}")

    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)