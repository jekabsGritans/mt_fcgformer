import json
import os
import threading
import time
from datetime import datetime

import numpy as np
import optuna
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template

# Load database configuration
load_dotenv()
OPTUNA_DB_URL="mysql+pymysql://user:hu4sie2Aiwee@192.168.6.5:3307/optuna"

# Study names from your optimization script
PHASE1_STUDY_NAME = "fcgformer-phase1-exploration"
PHASE2_STUDY_NAME = "fcgformer-phase2-exploitation" 
PHASE3_STUDY_NAME = "fcgformer-phase3-validation"

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
            "phase1": get_study_info(PHASE1_STUDY_NAME),
            "phase2": get_study_info(PHASE2_STUDY_NAME),
            "phase3": get_study_info(PHASE3_STUDY_NAME),
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
    # This endpoint can be expanded to include more detailed analysis
    best_trials = {}
    
    for phase_name, study_name in [("phase1", PHASE1_STUDY_NAME), 
                                 ("phase2", PHASE2_STUDY_NAME), 
                                 ("phase3", PHASE3_STUDY_NAME)]:
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

# Add right after your get_best_trials() function

@app.route('/api/param_importance/<phase>')
def get_param_importance(phase):
    """Get parameter importance visualization data"""
    study_name = {
        "phase1": PHASE1_STUDY_NAME,
        "phase2": PHASE2_STUDY_NAME,
        "phase3": PHASE3_STUDY_NAME
    }.get(phase)
    
    if not study_name:
        return jsonify({"error": "Invalid phase specified"})
    
    try:
        study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_URL)
        
        # Get all completed trials with values
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        
        if len(trials) < 5:
            return jsonify({"error": "Not enough completed trials for visualization"})
        
        # Get parameter values for all trials
        param_values = {}
        scores = []
        
        for trial in trials:
            scores.append(trial.value)
            for param, value in trial.params.items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)
        
        # Basic parameter importance using correlation
        importances = {}
        for param, values in param_values.items():
            # Convert categorical to numeric if needed
            numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(float(v))
                elif v == "True":
                    numeric_values.append(1.0)
                elif v == "False":
                    numeric_values.append(0.0)
                else:
                    # Skip parameters we can't easily convert
                    numeric_values = None
                    break
            
            # Calculate correlation if we have numeric values
            if numeric_values:
                try:
                    corr = abs(np.corrcoef(numeric_values, scores)[0, 1])
                    if not np.isnan(corr):
                        importances[param] = float(corr)
                except:
                    pass
        
        # Return the data
        return jsonify({
            "importances": importances,
            "trial_data": [
                {
                    "trial_id": t.number,
                    "value": t.value,
                    "params": t.params
                } for t in trials
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/parallel_coords/<phase>')
def get_parallel_coords_data(phase):
    """Get data for parallel coordinates plot"""
    study_name = {
        "phase1": PHASE1_STUDY_NAME,
        "phase2": PHASE2_STUDY_NAME,
        "phase3": PHASE3_STUDY_NAME
    }.get(phase)
    
    if not study_name:
        return jsonify({"error": "Invalid phase"})
    
    try:
        study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_URL)
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        
        if not trials:
            return jsonify({"error": "No completed trials"})
            
        # Get parameter names from first trial
        param_names = list(trials[0].params.keys())
        
        # Create dimensions for parallel coordinates
        dimensions = []
        for param in param_names:
            values = [t.params.get(param) for t in trials if param in t.params]
            
            # Handle different parameter types
            if all(isinstance(v, (int, float)) for v in values):
                dimensions.append({
                    "label": param,
                    "values": values,
                    "range": [min(values), max(values)]
                })
            elif all(v in [True, False, "True", "False"] for v in values):
                # Convert boolean to numeric
                numeric_values = [1 if v in [True, "True"] else 0 for v in values]
                dimensions.append({
                    "label": param,
                    "values": numeric_values,
                    "tickvals": [0, 1],
                    "ticktext": ["False", "True"]
                })
        
        # Add the objective value as the last dimension
        objective_values = [t.value for t in trials]
        dimensions.append({
            "label": "Score",
            "values": objective_values,
            "range": [min(objective_values), max(objective_values)]
        })
        
        return jsonify({
            "dimensions": dimensions,
            "trial_ids": [t.number for t in trials]
        })
    except Exception as e:
        return jsonify({"error": str(e)})


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
    <title>Optuna Hyperparameter Optimization Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
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
                        <div id="phase2-important" class="mt-3">
                            <h6>Important Parameters:</h6>
                            <pre id="phase2-important-json"></pre>
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
                        <div id="phase3-top" class="mt-3">
                            <h6>Top Parameters:</h6>
                            <pre id="phase3-top-json"></pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header bg-dark text-white">
                        Parameter Analysis
                    </div>
                    <div class="card-body">
                        <ul class="nav nav-tabs" id="paramTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="importance-tab" data-bs-toggle="tab" data-bs-target="#importance" type="button" role="tab">
                                    Parameter Importance
                                </button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="parallel-tab" data-bs-toggle="tab" data-bs-target="#parallel" type="button" role="tab">
                                    Parameter Relationships
                                </button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="distribution-tab" data-bs-toggle="tab" data-bs-target="#distribution" type="button" role="tab">
                                    Parameter Distributions
                                </button>
                            </li>
                        </ul>
                        
                        <div class="tab-content mt-3" id="paramTabsContent">
                            <div class="tab-pane fade show active" id="importance" role="tabpanel">
                                <div class="row mb-3">
                                    <div class="col-md-4">
                                        <select id="importance-phase" class="form-select">
                                            <option value="phase1">Phase 1: Exploration</option>
                                            <option value="phase2">Phase 2: Exploitation</option>
                                            <option value="phase3">Phase 3: Validation</option>
                                        </select>
                                    </div>
                                </div>
                                <div id="param-importance-plot"></div>
                            </div>
                            
                            <div class="tab-pane fade" id="parallel" role="tabpanel">
                                <div class="row mb-3">
                                    <div class="col-md-4">
                                        <select id="parallel-phase" class="form-select">
                                            <option value="phase1">Phase 1: Exploration</option>
                                            <option value="phase2">Phase 2: Exploitation</option>
                                            <option value="phase3">Phase 3: Validation</option>
                                        </select>
                                    </div>
                                </div>
                                <div id="parallel-coords-plot"></div>
                            </div>
                            
                            <div class="tab-pane fade" id="distribution" role="tabpanel">
                                <div class="row mb-3">
                                    <div class="col-md-4">
                                        <select id="dist-phase" class="form-select">
                                            <option value="phase1">Phase 1: Exploration</option>
                                            <option value="phase2">Phase 2: Exploitation</option>
                                            <option value="phase3">Phase 3: Validation</option>
                                        </select>
                                    </div>
                                    <div class="col-md-4">
                                        <select id="param-select" class="form-select">
                                            <option value="">Select parameter</option>
                                        </select>
                                    </div>
                                    <div class="col-md-4">
                                        <select id="top-n" class="form-select">
                                            <option value="5">Top 5 trials</option>
                                            <option value="10">Top 10 trials</option>
                                            <option value="all">All trials</option>
                                        </select>
                                    </div>
                                </div>
                                <div id="param-dist-plot"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <button id="refresh-btn" class="btn btn-primary refresh-btn">
        <i class="bi bi-arrow-repeat"></i> Refresh
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
            
            // Update phase-specific data
            if (phase === 'phase2' && data.user_attrs.important_params) {
                document.getElementById(`${phase}-important-json`).textContent = 
                    JSON.stringify(data.user_attrs.important_params, null, 2);
            }
            
            if (phase === 'phase3' && data.user_attrs.top_params) {
                document.getElementById(`${phase}-top-json`).textContent = 
                    JSON.stringify(data.user_attrs.top_params, null, 2);
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

        // Parameter importance visualization
        function updateParamImportance() {
            const phase = document.getElementById('importance-phase').value;
            
            fetch(`/api/param_importance/${phase}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('param-importance-plot').innerHTML = 
                            `<div class="alert alert-warning">${data.error}</div>`;
                        return;
                    }
                    
                    // Sort parameters by importance
                    const sortedImportance = Object.entries(data.importances)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10); // Show top 10 parameters
                    
                    const params = sortedImportance.map(d => d[0]);
                    const values = sortedImportance.map(d => d[1]);
                    
                    const plotData = [{
                        type: 'bar',
                        x: params,
                        y: values,
                        marker: {
                            color: values.map(v => `rgba(66, 133, 244, ${v})`),
                            line: {
                                color: 'rgb(8, 48, 107)',
                                width: 1.5
                            }
                        }
                    }];
                    
                    const layout = {
                        title: 'Parameter Importance',
                        xaxis: {
                            title: 'Parameter',
                            tickangle: 45
                        },
                        yaxis: {
                            title: 'Importance Score'
                        },
                        height: 500,
                        margin: { b: 150 }
                    };
                    
                    Plotly.newPlot('param-importance-plot', plotData, layout);
                });
        }

        // Parallel coordinates visualization
        function updateParallelCoords() {
            const phase = document.getElementById('parallel-phase').value;
            
            fetch(`/api/parallel_coords/${phase}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('parallel-coords-plot').innerHTML = 
                            `<div class="alert alert-warning">${data.error}</div>`;
                        return;
                    }
                    
                    // Create dimensions array for parallel coords
                    const dimensions = data.dimensions.map(d => ({
                        label: d.label,
                        values: d.values,
                        range: d.range,
                        tickvals: d.tickvals,
                        ticktext: d.ticktext
                    }));
                    
                    // Get the score dimension (last one)
                    const scoreDim = dimensions[dimensions.length - 1];
                    const scoreValues = scoreDim.values;
                    
                    // Calculate color scale based on score
                    const minScore = Math.min(...scoreValues);
                    const maxScore = Math.max(...scoreValues);
                    const normalizedScores = scoreValues.map(
                        s => (s - minScore) / (maxScore - minScore)
                    );
                    
                    const plotData = [{
                        type: 'parcoords',
                        dimensions: dimensions,
                        line: {
                            color: normalizedScores,
                            colorscale: 'Jet',
                            showscale: true
                        }
                    }];
                    
                    const layout = {
                        title: 'Parameter Relationships (Higher Score = Better)',
                        height: 600
                    };
                    
                    Plotly.newPlot('parallel-coords-plot', plotData, layout);
                });
        }

        // Parameter distribution visualization
        function populateParamSelect() {
            const phase = document.getElementById('dist-phase').value;
            
            fetch(`/api/param_importance/${phase}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error || !data.trial_data || data.trial_data.length === 0) {
                        return;
                    }
                    
                    // Get all parameter names
                    const params = Object.keys(data.trial_data[0].params);
                    
                    // Sort parameters by importance if available
                    let sortedParams = params;
                    if (data.importances) {
                        sortedParams = params.sort((a, b) => 
                            (data.importances[b] || 0) - (data.importances[a] || 0)
                        );
                    }
                    
                    // Update select options
                    const paramSelect = document.getElementById('param-select');
                    paramSelect.innerHTML = '<option value="">Select parameter</option>';
                    
                    sortedParams.forEach(param => {
                        const option = document.createElement('option');
                        option.value = param;
                        option.textContent = param;
                        paramSelect.appendChild(option);
                    });
                    
                    // Set first param if none selected
                    if (paramSelect.value === "") {
                        paramSelect.value = sortedParams[0] || "";
                    }
                    
                    // Update distribution plot
                    updateParamDistribution();
                });
        }

        function updateParamDistribution() {
            const phase = document.getElementById('dist-phase').value;
            const param = document.getElementById('param-select').value;
            const topN = document.getElementById('top-n').value;
            
            if (!param) return;
            
            fetch(`/api/param_importance/${phase}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error || !data.trial_data || data.trial_data.length === 0) {
                        document.getElementById('param-dist-plot').innerHTML = 
                            `<div class="alert alert-warning">No data available</div>`;
                        return;
                    }
                    
                    // Sort trials by score (descending)
                    const sortedTrials = [...data.trial_data].sort((a, b) => b.value - a.value);
                    
                    // Take top N trials or all
                    const trials = topN === 'all' ? 
                        sortedTrials : 
                        sortedTrials.slice(0, parseInt(topN));
                    
                    // Get parameter values and scores
                    const paramValues = trials.map(t => t.params[param]);
                    const scores = trials.map(t => t.value);
                    
                    // Decide on plot type based on parameter values
                    if (paramValues.every(v => typeof v === 'number')) {
                        // Numeric parameter - use scatter plot
                        const plotData = [{
                            type: 'scatter',
                            mode: 'markers',
                            x: paramValues,
                            y: scores,
                            marker: {
                                color: scores,
                                colorscale: 'Viridis',
                                size: 12
                            }
                        }];
                        
                        const layout = {
                            title: `${param} vs. Performance`,
                            xaxis: { title: param },
                            yaxis: { title: 'Score' },
                            height: 500
                        };
                        
                        Plotly.newPlot('param-dist-plot', plotData, layout);
                    } else {
                        // Categorical parameter - use box plots
                        // Group by parameter value
                        const groups = {};
                        trials.forEach(t => {
                            const val = String(t.params[param]);
                            if (!groups[val]) groups[val] = [];
                            groups[val].push(t.value);
                        });
                        
                        const plotData = Object.entries(groups).map(([val, scores]) => ({
                            type: 'box',
                            name: val,
                            y: scores,
                            boxpoints: 'all',
                            jitter: 0.3,
                            pointpos: 0
                        }));
                        
                        const layout = {
                            title: `Performance by ${param}`,
                            yaxis: { title: 'Score' },
                            height: 500
                        };
                        
                        Plotly.newPlot('param-dist-plot', plotData, layout);
                    }
                });
        }

        // Set up event handlers
        document.getElementById('importance-phase').addEventListener('change', updateParamImportance);
        document.getElementById('parallel-phase').addEventListener('change', updateParallelCoords);
        document.getElementById('dist-phase').addEventListener('change', populateParamSelect);
        document.getElementById('param-select').addEventListener('change', updateParamDistribution);
        document.getElementById('top-n').addEventListener('change', updateParamDistribution);

        // Initialize visualizations after page load
        document.addEventListener('DOMContentLoaded', function() {
            updateParamImportance();
            updateParallelCoords();
            populateParamSelect();
        });
    </script>
</body>
</html>
    ''')

if __name__ == '__main__':
    print("Starting Optuna Dashboard")
    print(f"Database URL: {OPTUNA_DB_URL}")
    print("Open http://127.0.0.1:5000 in your browser to view the dashboard")
    app.run(debug=True)