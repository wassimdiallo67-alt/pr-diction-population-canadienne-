"""API HTTP pour la prédiction de population canadienne."""

from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Charger le modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modele_final.pkl')
model_params = joblib.load(MODEL_PATH)

def growth_model(x, pop_base, taux_base, acceleration):
    """Modèle de croissance avec accélération après 2015."""
    t = x - 2000
    taux = taux_base + acceleration * np.maximum(0, t - 15)
    return pop_base * np.exp(taux * t / 100)

def predict_population(year):
    """Prédit la population pour une année donnée."""
    return growth_model(
        year,
        model_params['pop_base'],
        model_params['taux_base'],
        model_params['acceleration']
    )

@app.route('/', methods=['GET'])
def index():
    return '''<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Population Canada</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Inter', -apple-system, sans-serif; 
            background: #f8fafc;
            min-height: 100vh; 
            color: #1e293b;
        }
        
        .header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            padding: 60px 20px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; font-weight: 600; margin-bottom: 10px; }
        .header p { color: #94a3b8; font-size: 1.1em; }
        
        .container { max-width: 1000px; margin: 0 auto; padding: 40px 20px; }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
        }
        .card h2 { 
            font-size: 1.25em; 
            font-weight: 600; 
            margin-bottom: 20px;
            color: #0f172a;
        }
        
        .selector { 
            display: flex; 
            gap: 12px; 
            flex-wrap: wrap; 
            justify-content: center; 
            margin-bottom: 20px; 
        }
        .year-btn {
            padding: 14px 28px;
            font-size: 1em;
            font-weight: 500;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            cursor: pointer;
            background: white;
            color: #475569;
            transition: all 0.2s;
        }
        .year-btn:hover { 
            border-color: #3b82f6; 
            color: #3b82f6;
        }
        .year-btn.active { 
            background: #3b82f6; 
            border-color: #3b82f6;
            color: white; 
        }
        
        .custom-input {
            display: flex;
            gap: 12px;
            justify-content: center;
            align-items: center;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            margin-top: 20px;
        }
        .custom-input label { color: #64748b; font-size: 0.95em; }
        .custom-input input {
            padding: 12px 16px;
            font-size: 1em;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            width: 100px;
            text-align: center;
        }
        .custom-input input:focus { outline: none; border-color: #3b82f6; }
        .custom-input button {
            padding: 12px 24px;
            font-size: 1em;
            font-weight: 500;
            border: none;
            border-radius: 8px;
            background: #0f172a;
            color: white;
            cursor: pointer;
            transition: background 0.2s;
        }
        .custom-input button:hover { background: #1e293b; }
        
        #result { display: none; }
        
        .result-header {
            text-align: center;
            padding-bottom: 24px;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 24px;
        }
        .result-header h2 { font-size: 1.5em; color: #0f172a; }
        .result-header span { color: #3b82f6; }
        
        .stats { 
            display: grid; 
            grid-template-columns: repeat(4, 1fr); 
            gap: 16px; 
            margin-bottom: 30px; 
        }
        @media (max-width: 768px) {
            .stats { grid-template-columns: repeat(2, 1fr); }
        }
        .stat-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .stat-value { 
            font-size: 1.8em; 
            font-weight: 700; 
            color: #0f172a;
            margin-bottom: 4px;
        }
        .stat-value.blue { color: #3b82f6; }
        .stat-value.green { color: #10b981; }
        .stat-label { 
            color: #64748b; 
            font-size: 0.85em;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .progress-section { margin-bottom: 30px; }
        .progress-row { margin-bottom: 16px; }
        .progress-labels { 
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 6px;
            font-size: 0.9em;
            color: #64748b;
        }
        .progress-bar { 
            height: 8px; 
            background: #e2e8f0; 
            border-radius: 4px; 
            overflow: hidden; 
        }
        .progress-fill { 
            height: 100%; 
            border-radius: 4px; 
            transition: width 0.8s ease; 
        }
        .progress-fill.blue { background: #3b82f6; }
        .progress-fill.green { background: #10b981; }
        
        .analysis {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 24px;
        }
        .analysis h3 { 
            font-size: 1.1em; 
            font-weight: 600; 
            margin-bottom: 16px;
            color: #0f172a;
        }
        .analysis p { 
            line-height: 1.7; 
            color: #475569;
            margin-bottom: 12px;
        }
        .analysis p:last-child { margin-bottom: 0; }
        .analysis strong { color: #0f172a; }
        .analysis .note {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #e2e8f0;
            font-size: 0.9em;
            color: #64748b;
        }
        
        .model-info {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 24px;
        }
        .model-info h3 {
            color: #1e40af;
            font-size: 1em;
            margin-bottom: 10px;
        }
        .model-info p {
            color: #3b82f6;
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            color: #94a3b8;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Prédiction de la Population du Canada</h1>
    </div>
    
    
        
        <div class="card">
            <h2>Selectionner une année de prédiction</h2>
            <div class="selector">
                <button class="year-btn" onclick="predict(5, this)">2030 </button>
                <button class="year-btn" onclick="predict(10, this)">2035 </button>
                <button class="year-btn" onclick="predict(20, this)">2045 </button>
            </div>
            <div class="custom-input">
                <label> Perspective personnalisée:</label>
                <input type="number" id="customHorizon" min="1" max="75" value="15">
                <button onclick="predictCustom()">Calculer</button>
            </div>
        </div>
        
        <div id="result" class="card">
            <div class="result-header">
                <h2>Résultats pour l'année <span id="resultYear"></span></h2>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value blue" id="popValue">-</div>
                    <div class="stat-label">Population</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="hopValue">-</div>
                    <div class="stat-label">Hopitaux requis</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value green" id="newHopValue">-</div>
                    <div class="stat-label">A construire</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="growthValue">-</div>
                    <div class="stat-label">Croissance vs 2025</div>
                </div>
            </div>
            
            <div class="progress-section">
                <div class="progress-row">
                    <div class="progress-labels">
                        <span>Population 2025: 41.27M</span>
                        <span id="popBarLabel">-</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill blue" id="popBar" style="width:0%"></div>
                    </div>
                </div>
                <div class="progress-row">
                    <div class="progress-labels">
                        <span>Hopitaux 2025: 625</span>
                        <span id="hopBarLabel">-</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill green" id="hopBar" style="width:0%"></div>
                    </div>
                </div>
            </div>
            
            <div class="analysis">
                <h3>Analyse</h3>
                <div id="analysisText"></div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Source: Statistique Canada (1980-2025) | Ratio: 1 hopital pour 66 000 habitants | Modèle: GrowthModel</p>
    </div>
    
    <script>
        const RATIO_HOPITAL = 66000;
        const HOP_ACTUELS = 620;
        const POP_2025 = 41270000;
        
        async function predict(horizon, btn) {
            document.querySelectorAll('.year-btn').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({horizons: [horizon]})
                });
                const data = await response.json();
                
                if (data.predictions && data.predictions.length > 0) {
                    displayResult(data.predictions[0]);
                }
            } catch (e) {
                alert('Erreur: ' + e.message);
            }
        }
        
        function predictCustom() {
            const horizon = parseInt(document.getElementById('customHorizon').value);
            if (horizon >= 1 && horizon <= 75) {
                document.querySelectorAll('.year-btn').forEach(b => b.classList.remove('active'));
                predict(horizon, null);
            } else {
                alert('Veuillez entrer une perspective entre 1 et 75 ');
            }
        }
        
        function displayResult(pred) {
            document.getElementById('result').style.display = 'block';
            document.getElementById('resultYear').textContent = pred.year;
            
            const popM = (pred.prediction / 1000000).toFixed(2);
            const hopitaux = Math.round(pred.prediction / RATIO_HOPITAL);
            const nouveauxHop = hopitaux - HOP_ACTUELS;
            const growth = ((pred.prediction - POP_2025) / POP_2025 * 100).toFixed(1);
            
            document.getElementById('popValue').textContent = popM + 'M';
            document.getElementById('hopValue').textContent = hopitaux;
            document.getElementById('newHopValue').textContent = '+' + nouveauxHop;
            document.getElementById('growthValue').textContent = '+' + growth + '%';
            
            const popPercent = Math.min((pred.prediction / 80000000) * 100, 100);
            const hopPercent = Math.min((hopitaux / 1200) * 100, 100);
            document.getElementById('popBar').style.width = popPercent + '%';
            document.getElementById('hopBar').style.width = hopPercent + '%';
            document.getElementById('popBarLabel').textContent = popM + 'M';
            document.getElementById('hopBarLabel').textContent = hopitaux;
            
            const years = pred.horizon;
            const popIncrease = pred.prediction - POP_2025;
            const avgGrowth = Math.round(popIncrease / years);
            const hopPerYear = (nouveauxHop / years).toFixed(1);
            
            let analysis = '<p>Selon GrowthModel, la population du Canada atteindra <strong>' + popM + ' millions d\\'habitants</strong> en ' + pred.year + ', soit dans ' + years + ' ans.</p>';
            
            analysis += '<p>Cette évolution représente une augmentation de <strong>' + (popIncrease/1000000).toFixed(2) + ' millions de personnes</strong> par rapport à 2025, correspondant a une croissance annuelle moyenne de <strong>' + avgGrowth.toLocaleString() + ' habitants</strong>.</p>';
            
            analysis += '<p>Pour maintenir le ratio de couverture sanitaire (1 hopital pour 66 000 habitants), le système de santé canadien nécéssiterait <strong>' + hopitaux + ' hopitaux</strong>. Cela implique la construction de <strong>' + nouveauxHop + ' nouveaux établissements</strong>, soit environ <strong>' + hopPerYear + ' hopitaux par an</strong>.</p>';
            
            if (nouveauxHop > 100) {
                analysis += '<p>Ce volume de construction représente un défi majeur nécéssitant des investissements massifs en infrastructure et en formation de personnel médical.</p>';
            }
            
            if (years <= 5) {
                analysis += '<p class="note">Fiabilité: Elevée car Projections à court terme basées sur les tendances récentes.</p>';
            } else if (years <= 15) {
                analysis += '<p class="note">Fiabilité: Moyenne car Projections supposant le maintien des politiques actuelles.</p>';
            } else {
                analysis += '<p class="note">Fiabilité: Indicative car Projections à long terme sensibles aux changements de politiques.</p>';
            }
            
            document.getElementById('analysisText').innerHTML = analysis;
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>'''

@app.route('/predict/<horizons_str>', methods=['GET'])
def predict_get(horizons_str):
    try:
        horizons = [int(h.strip()) for h in horizons_str.split(',')]
    except ValueError:
        return jsonify({'error': 'Invalid horizons format'}), 400
    
    base_year = 2025
    predictions = []
    for h in horizons:
        if h < 0 or h > 100:
            return jsonify({'error': f'Horizon {h} out of range (0-100)'}), 400
        year = base_year + h
        pop = predict_population(year)
        predictions.append({'horizon': h, 'year': year, 'prediction': int(pop)})
    
    return jsonify({'predictions': predictions})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_type': model_params['type']
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    data = request.get_json()
    
    if data is None or 'horizons' not in data:
        return jsonify({'error': 'Missing required field: horizons'}), 400
    
    horizons = data['horizons']
    
    if not isinstance(horizons, list):
        return jsonify({'error': 'horizons must be a list of integers'}), 400
    
    for h in horizons:
        if not isinstance(h, int):
            return jsonify({'error': f'Invalid horizon value: {h}. Must be an integer.'}), 400
        if h < 0 or h > 100:
            return jsonify({'error': f'Horizon {h} out of range (0-100)'}), 400
    
    base_year = 2025
    predictions = []
    
    for h in horizons:
        year = base_year + h
        pop = predict_population(year)
        predictions.append({
            'horizon': h,
            'year': year,
            'prediction': int(pop)
        })
    
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
