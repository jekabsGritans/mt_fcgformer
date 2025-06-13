import json
import os
import threading
import time
from datetime import datetime

import optuna
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template

# Load database configuration
load_dotenv()
OPTUNA_DB_URL = "mysql+pymysql://user:hu4sie2Aiwee@192.168.6.5:3307/optuna"

# Base configuration - central point for derived names - MATCHES optimize.py
BASE_NAME = "mt_final_noaux"  # Change this to match optimize.py

# Study names derived from base name - CONSISTENT with optimize.py
STUDY_BASE = f"{BASE_NAME}"
STUDY_PHASE1 = f"{STUDY_BASE}-phase1-exploration"
STUDY_PHASE2 = f"{STUDY_BASE}-phase2-exploitation"
STUDY_PHASE3 = f"{STUDY_BASE}-phase3-validation"

# Initialize Flask app
app = Flask(__name__)

# Cache to store study data (avoid too many db calls)
cache = {
    "last_update": 0,
    "data": {},
    "update_lock": threading.Lock()
}

def get_study_info(study_name):
    """Get information about a study"""
    try:
        study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_URL)
        
        # Count trials by state
        trial_states = {
            "COMPLETE": 0,
            "PRUNED": 0,
            "RUNNING": 0,
            "FAILED": 0
        }
        
        for t in study.trials:
            state_name = str(t.state).split('.')[-1]
            if state_name in trial_states:
                trial_states[state_name] += 1
            else:
                trial_states[state_name] = 1
        
        # Get best trial details
        best_value = None
        best_params = {}
        if len(study.trials) > 0 and study.best_trial:
            best_value = study.best_value
            best_params = study.best_trial.params
            
        return {
            "name": study_name,
            "trial_states": trial_states,
            "total_trials": len(study.trials),
            "best_value": best_value,
            "best_params": best_params,
            "user_attrs": study.user_attrs
        }
    except Exception as e:
        return {
            "name": study_name,
            "error": str(e),
            "trial_states": {},
            "total_trials": 0
        }

def update_cache():
    """Update the cached data"""
    with cache["update_lock"]:
        # Only update if it's been more than 10 seconds
        if time.time() - cache["last_update"] < 10:
            return
        
        # Get data for all studies
        cache["data"] = {
            "phase1": get_study_info(STUDY_PHASE1),
            "phase2": get_study_info(STUDY_PHASE2),
            "phase3": get_study_info(STUDY_PHASE3),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Update last update timestamp
        cache["last_update"] = time.time()

@app.route('/')
def index():
    """Main dashboard view"""
    # Update cache in background
    threading.Thread(target=update_cache).start()
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint for dashboard data"""
    update_cache()
    return jsonify(cache["data"])

@app.route('/api/best_trials')
def get_best_trials():
    """Get best trials with detailed info"""
    best_trials = {}
    
    for phase_name, study_name in [("phase1", STUDY_PHASE1), 
                                 ("phase2", STUDY_PHASE2), 
                                 ("phase3", STUDY_PHASE3)]:
        try:
            study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_URL)
            if study.best_trial:
                best_trials[phase_name] = {
                    "value": study.best_value,
                    "params": study.best_trial.params,
                    "datetime": study.best_trial.datetime.strftime("%Y-%m-%d %H:%M:%S") 
                                if hasattr(study.best_trial, "datetime") else "Unknown"
                }
        except Exception:
            best_trials[phase_name] = {"error": "Not available"}
    
    return jsonify(best_trials)

@app.route('/best_runs')
def best_runs():
    best_runs = {}
    for phase, study_name in [
        (1, STUDY_PHASE1), 
        (2, STUDY_PHASE2), 
        (3, STUDY_PHASE3)
    ]:
        try:
            study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_URL)
            run_name = study.user_attrs.get('best_mlflow_run_name')
            val_f1 = study.user_attrs.get('best_val_f1')
            best_runs[f"phase{phase}"] = {
                "run_name": run_name,
                "val_f1": val_f1
            }
        except:
            best_runs[f"phase{phase}"] = {"error": "Study not found"}
    
    return render_template('best_runs.html', best_runs=best_runs)

# Create template directory
os.makedirs('templates', exist_ok=True)

# Create HTML template
with open('templates/dashboard.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FCGFormer Optimization Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        .phase-card {
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .card-header {
            font-weight: bold;
        }
        .refresh-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        .progress {
            height: 25px;
        }
        pre {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            max-height: 300px;
        }
    </style>
</head>
<body>
    <div class="container mt-4 mb-5">
        <h1 class="mb-4">FCGFormer Hyperparameter Optimization</h1>
        <p id="last-update" class="text-muted">Last updated: Loading...</p>
        
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        Overall Progress
                    </div>
                    <div class="card-body">
                        <div id="progress-overview"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-4">
                <div class="card phase-card">
                    <div class="card-header bg-info text-white">
                        Phase 1: Exploration
                    </div>
                    <div class="card-body">
                        <div id="phase1-chart"></div>
                        <div class="progress mt-3">
                            <div id="phase1-progress" class="progress-bar" role="progressbar"></div>
                        </div>
                        <p class="mt-2">Goal: 40 complete trials</p>
                        <hr>
                        <h5>Best Performance</h5>
                        <p id="phase1-best">Loading...</p>
                        <div id="phase1-params" class="mt-3">
                            <h6>Best Parameters:</h6>
                            <pre id="phase1-params-json"></pre>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card phase-card">
                    <div class="card-header bg-success text-white">
                        Phase 2: Exploitation
                    </div>
                    <div class="card-body">
                        <div id="phase2-chart"></div>
                        <div class="progress mt-3">
                            <div id="phase2-progress" class="progress-bar bg-success" role="progressbar"></div>
                        </div>
                        <p class="mt-2">Goal: 30 complete trials</p>
                        <hr>
                        <h5>Best Performance</h5>
                        <p id="phase2-best">Loading...</p>
                        <div id="phase2-params" class="mt-3">
                            <h6>Best Parameters:</h6>
                            <pre id="phase2-params-json"></pre>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card phase-card">
                    <div class="card-header bg-warning text-dark">
                        Phase 3: Validation
                    </div>
                    <div class="card-body">
                        <div id="phase3-chart"></div>
                        <div class="progress mt-3">
                            <div id="phase3-progress" class="progress-bar bg-warning" role="progressbar"></div>
                        </div>
                        <p class="mt-2">Goal: 15 complete trials</p>
                        <hr>
                        <h5>Best Performance</h5>
                        <p id="phase3-best">Loading...</p>
                        <div id="phase3-params" class="mt-3">
                            <h6>Best Parameters:</h6>
                            <pre id="phase3-params-json"></pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-dark text-white">
                        MLflow Best Runs
                    </div>
                    <div class="card-body">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Phase</th>
                                    <th>Best F1</th>
                                    <th>Run Name</th>
                                </tr>
                            </thead>
                            <tbody id="best-runs-body">
                                <tr><td colspan="3">Loading...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <button id="refresh-btn" class="btn btn-primary refresh-btn">
        Refresh
    </button>

    <script>
        // Initial data load
        fetchData();
        
        // Auto-refresh every 30 seconds
        setInterval(fetchData, 30000);
        
        // Manual refresh button
        document.getElementById('refresh-btn').addEventListener('click', fetchData);
        
        function fetchData() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    updateDashboard(data);
                    
                    // Get best runs
                    fetch('/api/best_trials')
                        .then(response => response.json())
                        .then(bestTrials => {
                            updateBestRuns(bestTrials);
                        });
                });
        }
        
        function updateDashboard(data) {
            // Update timestamp
            document.getElementById('last-update').textContent = 'Last updated: ' + data.timestamp;
            
            // Update phase 1
            updatePhaseCard('phase1', data.phase1, 40);
            
            // Update phase 2
            updatePhaseCard('phase2', data.phase2, 30);
            
            // Update phase 3
            updatePhaseCard('phase3', data.phase3, 15);
            
            // Update overall progress
            updateOverallProgress(data);
        }
        
        function updatePhaseCard(phase, data, goal) {
            if (data.error) {
                document.getElementById(`${phase}-best`).textContent = 'Study not found or error occurred';
                document.getElementById(`${phase}-progress`).style.width = '0%';
                document.getElementById(`${phase}-progress`).textContent = 'N/A';
                return;
            }
            
            // Update best value
            const bestElement = document.getElementById(`${phase}-best`);
            if (data.best_value !== null) {
                bestElement.textContent = `F1 Score: ${data.best_value.toFixed(4)}`;
            } else {
                bestElement.textContent = 'No successful trials yet';
            }
            
            // Update progress bar
            const complete = data.trial_states.COMPLETE || 0;
            const progressPercent = Math.min(100, Math.round((complete / goal) * 100));
            const progressBar = document.getElementById(`${phase}-progress`);
            progressBar.style.width = progressPercent + '%';
            progressBar.textContent = complete + '/' + goal + ' (' + progressPercent + '%)';
            
            // Update donut chart
            const chartData = [
                {
                    values: [
                        data.trial_states.COMPLETE || 0,
                        data.trial_states.PRUNED || 0,
                        data.trial_states.RUNNING || 0,
                        data.trial_states.FAILED || 0
                    ],
                    labels: ['Complete', 'Pruned', 'Running', 'Failed'],
                    type: 'pie',
                    hole: 0.4,
                    marker: {
                        colors: ['#28a745', '#ffc107', '#17a2b8', '#dc3545']
                    },
                    textinfo: "value"
                }
            ];
            
            const layout = {
                height: 200,
                margin: {l: 10, r: 10, t: 10, b: 10},
                showlegend: false
            };
            
            Plotly.newPlot(`${phase}-chart`, chartData, layout);
            
            // Update best parameters
            if (data.best_params && Object.keys(data.best_params).length > 0) {
                document.getElementById(`${phase}-params-json`).textContent = 
                    JSON.stringify(data.best_params, null, 2);
            } else {
                document.getElementById(`${phase}-params-json`).textContent = "No data yet";
            }
        }
        
        function updateOverallProgress(data) {
            // Calculate total trials across all phases
            const p1Complete = data.phase1.trial_states.COMPLETE || 0;
            const p2Complete = data.phase2.trial_states.COMPLETE || 0;
            const p3Complete = data.phase3.trial_states.COMPLETE || 0;
            
            const p1Running = data.phase1.trial_states.RUNNING || 0;
            const p2Running = data.phase2.trial_states.RUNNING || 0;
            const p3Running = data.phase3.trial_states.RUNNING || 0;
            
            const totalTrials = data.phase1.total_trials + data.phase2.total_trials + data.phase3.total_trials;
            const totalRunning = p1Running + p2Running + p3Running;
            
            // Approximate overall progress
            const p1Weight = 0.40; // Phase 1 is 40% of total progress
            const p2Weight = 0.35; // Phase 2 is 35% of total progress
            const p3Weight = 0.25; // Phase 3 is 25% of total progress
            
            const p1Progress = Math.min(1, p1Complete / 40) * p1Weight;
            const p2Progress = Math.min(1, p2Complete / 30) * p2Weight;
            const p3Progress = Math.min(1, p3Complete / 15) * p3Weight;
            
            const overallProgress = (p1Progress + p2Progress + p3Progress) * 100;
            
            // Create stacked bar chart
            const chartData = [
                {
                    x: ['Phase 1', 'Phase 2', 'Phase 3'],
                    y: [p1Complete, p2Complete, p3Complete],
                    name: 'Complete',
                    type: 'bar',
                    marker: {color: '#28a745'}
                },
                {
                    x: ['Phase 1', 'Phase 2', 'Phase 3'],
                    y: [p1Running, p2Running, p3Running],
                    name: 'Running',
                    type: 'bar',
                    marker: {color: '#17a2b8'}
                }
            ];
            
            const layout = {
                barmode: 'stack',
                height: 250,
                margin: {l: 50, r: 50, t: 20, b: 50},
                xaxis: {title: 'Phase'},
                yaxis: {title: 'Trial Count'},
                legend: {orientation: 'h', y: 1.1}
            };
            
            Plotly.newPlot('progress-overview', chartData, layout);
            
            // Display additional overall stats
            const phase1BestValue = data.phase1.best_value || 0;
            const phase2BestValue = data.phase2.best_value || 0;
            const phase3BestValue = data.phase3.best_value || 0;
            const overallBestValue = Math.max(phase1BestValue, phase2BestValue, phase3BestValue);
            
            const infoDiv = document.createElement('div');
            infoDiv.className = 'row mt-3';
            infoDiv.innerHTML = `
                <div class="col-md-3">
                    <div class="card text-white bg-primary">
                        <div class="card-body">
                            <h5 class="card-title">Overall Progress</h5>
                            <h2>${overallProgress.toFixed(1)}%</h2>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-success">
                        <div class="card-body">
                            <h5 class="card-title">Best F1 Score</h5>
                            <h2>${overallBestValue.toFixed(4)}</h2>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-info">
                        <div class="card-body">
                            <h5 class="card-title">Total Trials</h5>
                            <h2>${totalTrials}</h2>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-warning">
                        <div class="card-body">
                            <h5 class="card-title">Running Trials</h5>
                            <h2>${totalRunning}</h2>
                        </div>
                    </div>
                </div>
            `;
            
            const overviewElement = document.getElementById('progress-overview');
            const existingInfo = overviewElement.nextElementSibling;
            if (existingInfo && existingInfo.className === 'row mt-3') {
                existingInfo.remove();
            }
            overviewElement.insertAdjacentElement('afterend', infoDiv);
        }
        
        function updateBestRuns() {
            const tbodyElement = document.getElementById('best-runs-body');
            
            fetch('/best_runs')
                .then(response => response.text())
                .then(html => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const rows = [];
                    
                    for (let phase = 1; phase <= 3; phase++) {
                        const phaseKey = `phase${phase}`;
                        const phaseData = doc.querySelector(`#${phaseKey}-data`);
                        
                        if (phaseData) {
                            const runName = phaseData.dataset.runName || "N/A";
                            const valF1 = phaseData.dataset.valF1 || "N/A";
                            
                            rows.push(`
                                <tr>
                                    <td>Phase ${phase}</td>
                                    <td>${valF1}</td>
                                    <td>${runName}</td>
                                </tr>
                            `);
                        } else {
                            rows.push(`
                                <tr>
                                    <td>Phase ${phase}</td>
                                    <td colspan="2">Data not available</td>
                                </tr>
                            `);
                        }
                    }
                    
                    tbodyElement.innerHTML = rows.join('');
                })
                .catch(error => {
                    tbodyElement.innerHTML = `<tr><td colspan="3">Error loading data: ${error}</td></tr>`;
                });
        }
    </script>
</body>
</html>
    ''')

# Create template for best runs
with open('templates/best_runs.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Best MLflow Runs</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Best MLflow Runs</h1>
        
        {% for phase_key, phase_data in best_runs.items() %}
        <div id="{{ phase_key }}-data" 
             data-run-name="{{ phase_data.run_name|default('') }}"
             data-val-f1="{{ phase_data.val_f1|default('') }}">
            <h3>{{ phase_key|capitalize }}</h3>
            {% if phase_data.error %}
                <div class="alert alert-warning">{{ phase_data.error }}</div>
            {% else %}
                <p>Best Run: {{ phase_data.run_name|default('Not available') }}</p>
                <p>F1 Score: {{ phase_data.val_f1|default('Not available') }}</p>
            {% endif %}
        </div>
        <hr>
        {% endfor %}
        
        <a href="/" class="btn btn-primary">Back to Dashboard</a>
    </div>
</body>
</html>
    ''')

if __name__ == '__main__':
    print("Starting FCGFormer Optimization Dashboard")
    print(f"Database URL: {OPTUNA_DB_URL}")
    print("Open http://127.0.0.1:5000 in your browser to view the dashboard")
    app.run(debug=True)