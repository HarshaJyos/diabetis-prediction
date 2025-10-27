// server.js
const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(bodyParser.json());

let trees, scaler, leGender, means;

// -------------------------------------------------
// 1. Load Artifacts
// -------------------------------------------------
try {
  const modelPath = path.join(__dirname, 'diabetes_model.json');
  const scalerPath = path.join(__dirname, 'scaler.json');
  const genderPath = path.join(__dirname, 'le_gender.json');
  const meansPath = path.join(__dirname, 'column_means.json');

  const modelJson = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
  trees = modelJson?.learner?.gradient_booster?.model?.trees || [];

  if (!Array.isArray(trees) || trees.length === 0) {
    throw new Error('Invalid model JSON: No trees found');
  }

  scaler = JSON.parse(fs.readFileSync(scalerPath, 'utf8'));
  const genderJson = JSON.parse(fs.readFileSync(genderPath, 'utf8'));
  leGender = {};

  genderJson.classes.forEach((g, i) => {
    leGender[g.toUpperCase()] = i;
  });

  means = JSON.parse(fs.readFileSync(meansPath, 'utf8'));
} catch (err) {
  console.error('❌ Failed to load artifacts:', err.message);
  process.exit(1);
}

// -------------------------------------------------
// 2. Tree Walker (Single Tree Evaluation)
// -------------------------------------------------
function evaluateTree(tree, row) {
  const {
    base_weights,
    left_children,
    right_children,
    split_indices,
    split_conditions,
  } = tree;

  let nodeId = 0;

  while (true) {
    // Leaf node
    if (left_children[nodeId] === -1) {
      return base_weights[nodeId];
    }

    const featIdx = split_indices[nodeId];
    const threshold = split_conditions[nodeId];
    const goLeft = row[featIdx] < threshold;

    nodeId = goLeft ? left_children[nodeId] : right_children[nodeId];
  }
}

// -------------------------------------------------
// 3. Ensemble → Probability
// -------------------------------------------------
function predict(row) {
  let logOdds = 0;
  for (const tree of trees) {
    logOdds += evaluateTree(tree, row);
  }

  const prob = 1 / (1 + Math.exp(-logOdds));
  return { cls: prob > 0.5 ? 1 : 0, prob };
}

// -------------------------------------------------
// 4. Frontend (HTML UI)
// -------------------------------------------------
const HTML_TEMPLATE = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Diabetes Risk Prediction</title>
  <style>
    body {
      font-family: "Segoe UI", Arial, sans-serif;
      max-width: 700px;
      margin: 50px auto;
      background-color: #fafafa;
      color: #262730;
    }
    h1 { text-align: center; margin-bottom: 5px; }
    p { text-align: center; color: #555; }
    form {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      background: white;
      border: 1px solid #e0e0e0;
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    label {
      display: block;
      font-weight: 600;
      margin-bottom: 6px;
    }
    input, select {
      width: 100%;
      padding: 10px;
      border: 1px solid #ccc;
      border-radius: 6px;
      font-size: 15px;
      box-sizing: border-box;
    }
    button {
      grid-column: span 2;
      background: #0083ff;
      color: #fff;
      padding: 12px;
      font-size: 16px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.3s;
    }
    button:hover { background: #0072e6; }
    #spinner {
      display: none;
      margin: 20px auto;
      border: 4px solid #f3f3f3;
      border-top: 4px solid #3498db;
      border-radius: 50%;
      width: 25px;
      height: 25px;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    #result {
      margin-top: 25px;
      padding: 15px;
      text-align: center;
      border-radius: 8px;
      font-weight: bold;
      font-size: 16px;
    }
    .yes {
      background: #ffe6e6;
      color: #d60000;
      border: 1px solid #ffb3b3;
    }
    .no {
      background: #e6ffea;
      color: #007a00;
      border: 1px solid #b3ffbf;
    }
    footer {
      margin-top: 40px;
      text-align: center;
      color: #888;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <h1>Diabetes Risk Prediction</h1>
  <p>Enter your clinical values below.</p>

  <form id="predictForm">
    <div>
      <label for="age">Age</label>
      <input type="number" id="age" min="1" max="120" step="1" value="30" required />
    </div>
    <div>
      <label for="gender">Gender</label>
      <select id="gender" required>
        ${Object.keys(leGender)
          .map((g) => `<option value="${g}">${g.toUpperCase()}</option>`)
          .join('')}
      </select>
    </div>
    <div>
      <label for="bmi">BMI</label>
      <input type="number" id="bmi" min="10.0" max="60.0" step="0.1" value="25.0" required />
    </div>
    <div>
      <label for="hba1c">HbA1c (%)</label>
      <input type="number" id="hba1c" min="0.0" max="20.0" step="0.1" value="5.0" required />
    </div>
    <div>
      <label for="chol">Cholesterol (mg/dL)</label>
      <input type="number" id="chol" min="0" max="600.0" step="0.1" value="180.0" required />
    </div>
    <div>
      <label for="tg">Triglycerides (TG)</label>
      <input type="number" id="tg" min="0.0" max="2000.0" step="0.1" value="150.0" required />
    </div>
    <button type="submit">Predict Diabetes Risk</button>
  </form>

  <div id="spinner" class="spinner"></div>
  <div id="result"></div>

  <footer>
    Model: XGBoost (99% accuracy) | Uses 11 clinical features | Built with Streamlit
  </footer>

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
        tg: parseFloat(document.getElementById('tg').value),
      };

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });

        const json = await res.json();
        if (json.error) throw new Error(json.error);

        resultDiv.textContent = \`\${json.message} (Confidence: \${json.confidence}%)\`;
        resultDiv.className = json.result === 'Yes' ? 'yes' : 'no';
      } catch (err) {
        resultDiv.textContent = 'Error: ' + err.message;
      } finally {
        spinner.style.display = 'none';
      }
    });
  </script>
</body>
</html>
`;

// -------------------------------------------------
// 5. Serve HTML
// -------------------------------------------------
app.get('/', (req, res) => res.send(HTML_TEMPLATE));

// -------------------------------------------------
// 6. /predict Endpoint
// -------------------------------------------------
app.post('/predict', (req, res) => {
  try {
    const { age, gender, hba1c, bmi, chol, tg } = req.body;
    if (![age, gender, hba1c, bmi, chol, tg].every((v) => v !== undefined)) {
      throw new Error('Missing input fields');
    }

    const gEnc = leGender[gender.toUpperCase()] ?? 0;

    const rawRow = [
      gEnc, age,
      means.Urea, means.Cr,
      hba1c,
      chol, tg,
      means.HDL, means.LDL,
      means.VLDL, bmi,
    ];

    const scaledRow = rawRow.map((v, i) =>
      (v - scaler.mean[i]) / scaler.scale[i]
    );

    const { cls, prob } = predict(scaledRow);
    const result = cls === 1 ? 'Yes' : 'No';
    const confidence = (prob * 100).toFixed(1);

    res.json({
      result,
      message: `Diabetes Risk: ${result}`,
      confidence,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// -------------------------------------------------
// 7. Start Server
// -------------------------------------------------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log(`✅ Server running at http://localhost:${PORT}`)
);
