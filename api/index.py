# api/index.py (for Vercel serverless)
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load artifacts
@ app.before_first_request
def load_models():
    global model, scaler, le_gender, means
    model = joblib.load('../diabetes_model.pkl')
    scaler = joblib.load('../scaler.pkl')
    le_gender = joblib.load('../le_gender.pkl')
    means = joblib.load('../column_means.pkl')

# HTML template for the webpage
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background: #f9f9f9; }
        h1 { text-align: center; color: #333; }
        form { display: grid; gap: 15px; }
        label { font-weight: bold; }
        input, select { padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
        button { background: #4CAF50; color: white; padding: 10px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #45a049; }
        #result { margin-top: 20px; padding: 15px; border-radius: 5px; text-align: center; font-weight: bold; }
        .yes { background: #ffdddd; color: #d00; }
        .no { background: #ddffdd; color: #080; }
        .spinner { display: none; margin: 20px auto; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <h1>Diabetes Risk Prediction (XGBoost)</h1>
    <form id="predictForm">
        <label for="age">Age</label>
        <input type="number" id="age" min="1" max="120" value="30" required>

        <label for="gender">Gender</label>
        <select id="gender" required>
            {% for opt in gender_options %}
            <option value="{{ opt }}">{{ opt }}</option>
            {% endfor %}
        </select>

        <label for="hba1c">HbA1c (%)</label>
        <input type="number" id="hba1c" min="0" max="20" step="0.1" value="5.0" required>

        <label for="bmi">BMI</label>
        <input type="number" id="bmi" min="10.0" max="60.0" step="0.1" value="25.0" required>

        <label for="chol">Cholesterol (mg/dL)</label>
        <input type="number" id="chol" min="50.0" max="600.0" step="1.0" value="180.0" required>

        <label for="tg">Triglycerides (TG)</label>
        <input type="number" id="tg" min="10.0" max="2000.0" step="1.0" value="150.0" required>

        <button type="submit">Predict</button>
    </form>
    <div id="spinner" class="spinner"></div>
    <div id="result"></div>

    <script>
        const form = document.getElementById('predictForm');
        const spinner = document.getElementById('spinner');
        const resultDiv = document.getElementById('result');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            spinner.style.display = 'block';
            resultDiv.textContent = '';
            resultDiv.className = '';

            const data = {
                age: parseInt(document.getElementById('age').value),
                gender: document.getElementById('gender').value,
                hba1c: parseFloat(document.getElementById('hba1c').value),
                bmi: parseFloat(document.getElementById('bmi').value),
                chol: parseFloat(document.getElementById('chol').value),
                tg: parseFloat(document.getElementById('tg').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const json = await response.json();
                if (json.error) {
                    resultDiv.textContent = json.error;
                } else {
                    resultDiv.textContent = `${json.message} (Confidence: ${json.confidence}%)`;
                    resultDiv.className = json.result === 'Yes' ? 'yes' : 'no';
                }
            } catch (err) {
                resultDiv.textContent = `Error: ${err.message}`;
            } finally {
                spinner.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    gender_options = [str(g).strip().upper() for g in le_gender.classes_]
    return render_template_string(HTML_TEMPLATE, gender_options=gender_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        age = data['age']
        gender = data['gender']
        hba1c = data['hba1c']
        bmi = data['bmi']
        chol = data['chol']
        tg = data['tg']

        g_enc = le_gender.transform([gender])[0]

        row = np.array([[
            g_enc, age,
            means.get('Urea', 0), means.get('Cr', 0),
            hba1c,
            chol, tg,
            means.get('HDL', 0), means.get('LDL', 0),
            means.get('VLDL', 0), bmi
        ]])

        row_scaled = scaler.transform(row)
        pred = model.predict(row_scaled)[0]
        prob = model.predict_proba(row_scaled)[0][pred] * 100
        result = "Yes" if pred == 1 else "No"
        message = f"Diabetes Risk: {result}"
        confidence = f"{prob:.1f}"

        return jsonify({"result": result, "message": message, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel handler
def handler(event, context):
    from werkzeug.wrappers import Request, Response
    from werkzeug.serving import run_simple
    req = Request(event)
    with app.app_context():
        response = app.full_dispatch_request()
    return response.get_response()

if __name__ == '__main__':
    app.run(debug=True)