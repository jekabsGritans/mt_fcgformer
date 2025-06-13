import glob
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

import mlflow
import optuna
from dotenv import load_dotenv

from optimize_suggest_params import suggest_parameters
from utils.mlflow_utils import configure_mlflow_auth

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optuna_optimization.log')
    ]
)
logger = logging.getLogger("optuna_optimization")

# Generate a unique worker ID
WORKER_ID = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

# Base configuration - central point for derived names
BASE_NAME = "mt_debug_aux"
MODEL_NAME = "mt_fcgformer"  # Model architecture name
DATASET_ID = "157d4b53c95f4af88ee86fbcc319bce2"

# MLflow experiment names
MLFLOW_EXP_OPTUNA = f"{BASE_NAME}-optuna"  # Optuna trials tracking
MLFLOW_EXP_MONITOR = f"{BASE_NAME}-monitoring"  # Worker monitoring
MLFLOW_EXP_TRAINING = f"{BASE_NAME}-training"  # For main.py training runs

# Optuna study names
STUDY_BASE = f"{BASE_NAME}"
STUDY_PHASE1 = f"{STUDY_BASE}-phase1-exploration"
STUDY_PHASE2 = f"{STUDY_BASE}-phase2-exploitation" 
STUDY_PHASE3 = f"{STUDY_BASE}-phase3-validation"

# Map trial IDs to MLflow run IDs
RUN_ID_MAP = {}

# Set up Optuna database URL
load_dotenv()
OPTUNA_DB_URL = os.getenv("OPTUNA_DB_URL")

# Minimum number of epochs before pruning
MIN_EPOCHS = 2

# if DONT_OPTIMIZE env variable is set to true, exit script
if os.getenv("DONT_OPTIMIZE", "false").lower() == "true":
    logger.info("Skipping optimization as DONT_OPTIMIZE is set to true")
    sys.exit(0)

# Global variable to track the current process
current_process = None

def update_job_status(status, error=None):
    """Report job status to MLflow without creating redundant runs"""
    try:
        mlflow.set_experiment(MLFLOW_EXP_MONITOR)
        
        # Use consistent run ID based on worker ID
        run_key = f"worker_{WORKER_ID}"
        
        # Find existing run or create a new one
        if run_key in RUN_ID_MAP:
            with mlflow.start_run(run_id=RUN_ID_MAP[run_key]):
                mlflow.set_tag("status", status)
                mlflow.set_tag("last_update", time.strftime("%Y-%m-%d %H:%M:%S"))
                mlflow.set_tag("phase", get_current_phase())
                if error:
                    mlflow.set_tag("error", str(error))
        else:
            # Create new run
            with mlflow.start_run(run_name=run_key) as run:
                mlflow.set_tag("hostname", socket.gethostname())
                mlflow.set_tag("worker_id", WORKER_ID)
                mlflow.set_tag("status", status)
                mlflow.set_tag("phase", get_current_phase())
                mlflow.set_tag("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                if error:
                    mlflow.set_tag("error", str(error))
                
                # Log IP address for remote access
                try:
                    ip = subprocess.check_output("curl -s ifconfig.me", shell=True).decode().strip()
                    mlflow.set_tag("ip_address", ip)
                except:
                    pass
                
                # Save run ID for future updates
                RUN_ID_MAP[run_key] = run.info.run_id
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")

def gracefully_terminate_process(process, timeout=30):
    """Send SIGTERM signal and wait for process to finish gracefully"""
    if process and process.poll() is None:
        logger.info(f"Sending SIGTERM to process {process.pid}")
        process.send_signal(signal.SIGTERM)
        
        # Wait for the process to terminate gracefully
        try:
            process.wait(timeout=timeout)
            logger.info("Process terminated gracefully")
            return True
        except subprocess.TimeoutExpired:
            logger.warning(f"Process did not terminate within {timeout}s, forcing termination")
            process.kill()
            return False
    return True


def handle_sigterm(signum, frame):
    """Handle termination signal"""
    update_job_status("terminated")
    global current_process
    logger.info("Received termination signal")
    if current_process and current_process.poll() is None:
        gracefully_terminate_process(current_process)
    sys.exit(0)

def set_study_user_attr_safe(study, key, value):
    """Thread-safe way to set user attributes in a distributed environment"""
    max_retries = 5
    retry_delay = 0.5  # Start with 0.5 second delay
    
    for attempt in range(max_retries):
        try:
            study.set_user_attr(key, value)
            return True
        except Exception as e:
            if "locked" in str(e).lower() or "concurrent" in str(e).lower():
                # This is likely a database lock error
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Database lock detected when setting '{key}'. "
                              f"Retrying in {wait_time:.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # Different error
                logger.error(f"Error setting user attribute '{key}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return False
    
    logger.error(f"Failed to set user attribute '{key}' after {max_retries} attempts")
    return False

def get_parameter_importance(study):
    try:
        importances = optuna.importance.get_param_importances(study)
        
        # Convert to sorted list of (param, importance) tuples
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Parameter importance calculated: {sorted_importances[:5]}")
        
        return dict(importances)
    except Exception as e:
        logger.error(f"Error calculating parameter importance: {e}")
        return {}

def monitor_training_process(current_process, training_log, metrics_file, trial, phase):
    """Monitor training process and check metrics periodically"""
    best_val_f1 = 0
    last_epoch = 0
    no_improvement_count = 0
    
    # Adjust patience and early stopping based on phase
    early_stop_epochs = 10 if phase == 1 else 20 if phase == 2 else 30
    patience_threshold = 5 if phase == 1 else 8 if phase == 2 else 12
    min_f1_threshold = 0.4 if phase == 1 else 0.5 if phase == 2 else 0.6
    
    while current_process.poll() is None:
        time.sleep(10)  # Check metrics every 10 seconds
        
        # Check if metrics file exists and read it
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                current_epoch = metrics_data.get("epoch", 0)
                val_f1 = metrics_data.get("val/fg/weighted_f1", 0)
                val_loss = metrics_data.get("val/loss/total", float('inf'))
                
                # Only log if epoch changed
                if current_epoch > last_epoch:
                    logger.info(f"Trial {trial.number} P{phase} - Epoch {current_epoch}: val_f1={val_f1:.4f}, val_loss={val_loss:.4f}")
                    last_epoch = current_epoch
                    
                    # Report to Optuna for pruning
                    if current_epoch >= MIN_EPOCHS:
                        trial.report(val_f1, current_epoch)
                        
                        # Track best F1 score
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                        
                        # Check if Optuna decides to prune
                        if trial.should_prune():
                            logger.info(f"Trial {trial.number} P{phase} pruned by Optuna at epoch {current_epoch}")
                            gracefully_terminate_process(current_process)
                            return val_f1
                        
                        # Custom early stopping for bad runs - stricter in later phases
                        if current_epoch >= early_stop_epochs and val_f1 < min_f1_threshold:
                            logger.info(f"Trial {trial.number} P{phase} stopped early due to poor performance")
                            gracefully_terminate_process(current_process)
                            return val_f1
                        
                        # Custom early stopping if no improvement for several epochs
                        if no_improvement_count >= patience_threshold and current_epoch >= early_stop_epochs:
                            logger.info(f"Trial {trial.number} P{phase} stopped early due to no improvement")
                            gracefully_terminate_process(current_process)
                            return best_val_f1
                            
            except (json.JSONDecodeError, IOError):
                # File might be being written to
                pass
    
    # Process completed naturally
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                final_metrics = json.load(f)
            final_val_f1 = final_metrics.get("val/fg/weighted_f1", 0)
            logger.info(f"Trial {trial.number} P{phase} completed with final val_f1={final_val_f1:.4f}")
            return final_val_f1
        except (json.JSONDecodeError, IOError):
            logger.warning(f"Trial {trial.number} P{phase} - Could not read final metrics file")
            return best_val_f1  # Return the best observed value
    
    logger.warning(f"Trial {trial.number} P{phase} - No metrics file available")
    return best_val_f1

def build_training_command(params, phase, metrics_file, trial_number):
    """Build command array for main.py training script"""

    trial_run_name = f"{STUDY_BASE}-p{phase}-t{trial_number}"

    # Extract individual parameters
    lr = params["lr"]
    batch_size = params["batch_size"]
    weight_decay = params["weight_decay"]
    warmup_steps = params["warmup_steps"]
    scheduler_t0 = params["scheduler_t0"]
    scheduler_tmult = params["scheduler_tmult"]
    
    patch_size = params["patch_size"]
    embed_dim = params["embed_dim"]
    num_layers = params["num_layers"]
    expansion_factor = params["expansion_factor"]
    n_heads = params["n_heads"]
    dropout_p = params["dropout_p"]
    
    use_noise = params["use_noise"]
    use_mask = params["use_mask"]
    use_shiftud = params["use_shiftud"]
    use_shiftlr = params["use_shiftlr"]
    use_revert = params["use_revert"]

    # Extract auxiliary loss parameters
    initial_aux_bool_weight = params["initial_aux_bool_weight"]
    initial_aux_float_weight = params["initial_aux_float_weight"]
    aux_epochs = params["aux_epochs"]
    
    # Extract dataset weights
    nist_lser_weight = params["nist_lser_weight"]
    chemmotion_weight = params["chemmotion_weight"]
    chemmotion_lser_weight = params["chemmotion_lser_weight"]
    graphformer_weight = params["graphformer_weight"]
    graphformer_lser_weight = params["graphformer_lser_weight"]
    
    # Build command with base parameters
    cmd = [
        sys.executable, "main.py",
        "mode=train",
        "device=cuda:0",
        f"experiment_name={MLFLOW_EXP_TRAINING}_phase{phase}",
        f"dataset_id={DATASET_ID}",
        f"model={MODEL_NAME}",
        f"run_name={trial_run_name}",
        
        # Trainer params
        f"trainer.lr={lr}",
        f"trainer.batch_size={batch_size}",
        f"trainer.weight_decay={weight_decay}",
        f"trainer.warmup_steps={warmup_steps}",
        f"trainer.scheduler_t0={scheduler_t0}",
        f"trainer.scheduler_tmult={scheduler_tmult}",
        
        # Auxiliary loss parameters
        f"trainer.initial_aux_bool_weight={initial_aux_bool_weight}",
        f"trainer.initial_aux_float_weight={initial_aux_float_weight}",
        f"trainer.aux_epochs={aux_epochs}",
        
        # Model params
        f"model.patch_size={patch_size}",
        f"model.embed_dim={embed_dim}",
        f"model.num_layers={num_layers}",
        f"model.expansion_factor={expansion_factor}",
        f"model.n_heads={n_heads}",
        f"model.dropout_p={dropout_p}",
        
        # Transform control flags
        f"use_noise={use_noise}",
        f"use_mask={use_mask}",
        f"use_shiftud={use_shiftud}",
        f"use_shiftlr={use_shiftlr}",
        f"use_revert={use_revert}",
        
        # Fixed parameters - adjust based on phase
        f"trainer.epochs={30 if phase == 3 else 20 if phase == 2 else 10}",
        f"trainer.patience={20 if phase == 3 else 30 if phase == 2 else 40}",
        f"metric_output_file={metrics_file}",
        f"skip_checkpoints={phase != 3}",
        
        # Dataset weights
        "nist_weight=1.0",  # Always fixed at 1.0
        f"nist_lser_weight={nist_lser_weight}",
        f"chemmotion_weight={chemmotion_weight}",
        f"chemmotion_lser_weight={chemmotion_lser_weight}",
        f"graphformer_weight={graphformer_weight}",
        f"graphformer_lser_weight={graphformer_lser_weight}",
    ]
    
    return cmd, trial_run_name

def track_best_run_in_study(study):
    """Save the best run name in the study user attributes"""
    if not study.best_trial:
        return False
    
    # Get MLflow run name from best trial
    best_run_name = study.best_trial.user_attrs.get('mlflow_run_name')
    best_run_id = study.best_trial.user_attrs.get('mlflow_run_id')
    
    if best_run_name:
        set_study_user_attr_safe(study, 'best_mlflow_run_name', best_run_name)
        if best_run_id:
            set_study_user_attr_safe(study, 'best_mlflow_run_id', best_run_id)
        set_study_user_attr_safe(study, 'best_val_f1', study.best_value)
        return True
    return False

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    global current_process

    # Clean up old trials if too many exist
    trial_dirs = glob.glob("./trials/phase*_trial_*")
    if len(trial_dirs) > 20:  # Keep only most recent 20 trials
        for old_dir in sorted(trial_dirs, key=os.path.getctime)[:len(trial_dirs)-20]:
            try:
                shutil.rmtree(old_dir)
            except:
                pass  # Ignore errors

    # Get current phase from study user attrs
    phase = trial.study.user_attrs.get('phase', 1)
    study_name = trial.study.study_name
    
    # Get parameters for current phase
    params = suggest_parameters(trial, phase)
    
    # Create a unique directory for this trial's files
    trial_metrics_dir = os.path.join("./trials", f"phase{phase}_trial_{trial.number}")
    os.makedirs(trial_metrics_dir, exist_ok=True)
    metrics_file = os.path.join(trial_metrics_dir, "metrics.json")
    training_log = os.path.join(trial_metrics_dir, "training.log")
    
    # Build command for training and get unique run name
    cmd, trial_run_name = build_training_command(params, phase, metrics_file, trial.number)
    
    # Log detailed trial information
    transform_info = f"Transforms: noise={params['use_noise']}, mask={params['use_mask']}, " \
                     f"shiftUD={params['use_shiftud']}, shiftLR={params['use_shiftlr']}, " \
                     f"revert={params['use_revert']}"
    model_info = f"Model: layers={params['num_layers']}, heads={params['n_heads']}, " \
                 f"embed_dim={params['embed_dim']}, patch={params['patch_size']}"
    phase_info = f"Phase {phase}"
    logger.info(f"Starting trial {trial.number} ({phase_info}) with {model_info}, {transform_info}")
    
    # Log trial to MLflow directly for better tracking
    try:
        mlflow.set_experiment(MLFLOW_EXP_OPTUNA)
        with mlflow.start_run(run_name=trial_run_name) as run:
            # Log study and trial information
            mlflow.set_tag("optuna_study", study_name)
            mlflow.set_tag("optuna_phase", phase)
            mlflow.set_tag("optuna_trial", trial.number)
            
            # Log all parameters
            mlflow.log_params(params)
            
            # Store MLflow run info in trial user attributes for later reference
            trial.set_user_attr('mlflow_run_id', run.info.run_id)
            trial.set_user_attr('mlflow_run_name', trial_run_name)
            
            logger.info(f"Linked trial {trial.number} to MLflow run: {trial_run_name}")
    except Exception as e:
        logger.warning(f"Failed to log trial to MLflow: {e}")
    
    try:
        # Start the training process with output redirected to log file
        with open(training_log, 'w') as log_file:
            current_process = subprocess.Popen(
                cmd, 
                stdout=log_file, 
                stderr=subprocess.STDOUT
            )
        
        # Monitor the training process and handle early stopping
        val_f1 = monitor_training_process(current_process, training_log, metrics_file, trial, phase)
        
        # If this is potentially a best run, mark it in the trial
        if val_f1 > 0.7:  # Reasonably good F1 score
            trial.set_user_attr('potential_best', True)
            logger.info(f"Trial {trial.number} achieved high F1: {val_f1:.4f}")
            
            # Try to update MLflow with final result
            try:
                run_id = trial.user_attrs.get('mlflow_run_id')
                if run_id:
                    with mlflow.start_run(run_id=run_id):
                        mlflow.log_metric("val_f1_final", val_f1)
                        mlflow.set_tag("trial_complete", "true")
            except Exception as e:
                logger.warning(f"Failed to update MLflow run with final result: {e}")
        
        return val_f1
        
    except Exception as e:
        logger.error(f"Error in trial {trial.number} P{phase}: {e}")
        if current_process and current_process.poll() is None:
            gracefully_terminate_process(current_process)
        
        # Log the error to MLflow if possible
        try:
            run_id = trial.user_attrs.get('mlflow_run_id')
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.set_tag("error", str(e))
        except:
            pass
            
        return 0.0
        
    finally:
        current_process = None

def run_phase1(n_trials=30):
    """Phase 1: Wide exploration with per-trial phase advancement check"""
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    logger.info(f"Phase 1: Starting EXPLORATION with up to {n_trials} trials")
    
    # Create phase 1 study with RandomSampler for better exploration
    study = optuna.create_study(
        study_name=STUDY_PHASE1,
        storage=OPTUNA_DB_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Set phase metadata
    set_study_user_attr_safe(study, "phase", 1)
    
    try:
        # Run optimization ONE TRIAL AT A TIME and check advancement condition after each
        for i in range(n_trials):
            # Check if we already have enough trials BEFORE running the next one
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            if completed_trials >= 40:  # Phase 1 threshold
                logger.info(f"Phase 1 already has {completed_trials} trials, stopping early after {i} trials from this worker")
                break
            
            # Run just one trial
            study.optimize(objective, n_trials=1)
            
            # Re-check after this trial if we've hit the threshold
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            logger.info(f"Phase 1 now has {completed_trials}/40 trials completed")
        
        # Analyze parameter importance
        importance_dict = get_parameter_importance(study)
        
        # Extract important parameters (top 8 or threshold-based)
        if importance_dict:
            important_params = {}
            for param, importance in importance_dict.items():
                # Only include parameters with at least 1% importance
                if importance >= 0.01:
                    important_params[param] = importance
            
            # If we have too many important parameters, limit to top 8
            if len(important_params) > 8:
                important_params = dict(sorted(important_params.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:8])
            best_params = study.best_params

            track_best_run_in_study(study)
            
            return study.best_value, best_params, important_params
            
    except KeyboardInterrupt:
        logger.info("Phase 1 interrupted by user")
    except Exception as e:
        logger.error(f"Error during Phase 1: {e}")
        
    # If we reach here, something went wrong
    return 0.0, {}, {}

def run_phase2(best_phase1_params, important_params, n_trials=20):
    """Phase 2: Exploitation with per-trial phase advancement check"""
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    logger.info(f"Phase 2: Starting EXPLOITATION with up to {n_trials} trials")
    logger.info(f"Using top parameters: {list(important_params.keys())}")
    
    # Create phase 2 study with TPESampler for exploitation
    study = optuna.create_study(
        study_name=STUDY_PHASE2,
        storage=OPTUNA_DB_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Set phase metadata and carry forward important parameters
    set_study_user_attr_safe(study, "phase", 2)
    set_study_user_attr_safe(study, "best_phase1_params", best_phase1_params)
    set_study_user_attr_safe(study, "important_params", important_params)
    
    try:
        # Start with the best parameters from phase 1
        study.enqueue_trial(best_phase1_params)
        
        # Run optimization ONE TRIAL AT A TIME
        for i in range(n_trials):
            # Check if we already have enough trials BEFORE running the next one
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            if completed_trials >= 30:  # Phase 2 threshold
                logger.info(f"Phase 2 already has {completed_trials} trials, stopping early after {i} trials from this worker")
                break
            
            # Run just one trial
            study.optimize(objective, n_trials=1)
            
            # Re-check after this trial if we've hit the threshold
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            logger.info(f"Phase 2 now has {completed_trials}/30 trials completed")
        
        # Identify top parameters for phase 3
        importance_dict = get_parameter_importance(study)
        
        # Extract top parameters (only top 3-5)
        if importance_dict:
            # Get top 5 parameters by importance
            top_params = list(dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:5]).keys())
            
            logger.info(f"Top parameters for fine-tuning: {top_params}")

            best_params = study.best_params

            track_best_run_in_study(study)
            
            return study.best_value, best_params, top_params
            
    except KeyboardInterrupt:
        logger.info("Phase 2 interrupted by user")
    except Exception as e:
        logger.error(f"Error during Phase 2: {e}")
        
    # If we reach here, something went wrong
    return 0.0, {}, []


def run_phase3(best_phase2_params, top_params, n_trials=10):
    """Phase 3: Final validation with per-trial phase advancement check"""
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    logger.info(f"Phase 3: Starting VALIDATION with up to {n_trials} trials")
    logger.info(f"Fine-tuning parameters: {top_params}")
    
    # Create phase 3 study with focused TPESampler
    study = optuna.create_study(
        study_name=STUDY_PHASE3,
        storage=OPTUNA_DB_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Set phase metadata and carry forward best parameters
    set_study_user_attr_safe(study, "phase", 3)
    set_study_user_attr_safe(study, "best_phase2_params", best_phase2_params)
    set_study_user_attr_safe(study, "top_params", top_params)
    
    try:
        # Start with the best parameters from phase 2
        study.enqueue_trial(best_phase2_params)
        
        # Run optimization ONE TRIAL AT A TIME
        for i in range(n_trials):
            # Check if we already have enough trials BEFORE running the next one
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            if completed_trials >= 15:  # Phase 3 threshold
                logger.info(f"Phase 3 already has {completed_trials} trials, stopping early after {i} trials from this worker")
                break
            
            # Run just one trial
            study.optimize(objective, n_trials=1)
            
            # Re-check after this trial if we've hit the threshold
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            logger.info(f"Phase 3 now has {completed_trials}/15 trials completed")
        
        # Extract final best parameters
        final_best_params = study.best_params

        track_best_run_in_study(study)
        
        return study.best_value, final_best_params
            
    except KeyboardInterrupt:
        logger.info("Phase 3 interrupted by user")
    except Exception as e:
        logger.error(f"Error during Phase 3: {e}")
        
    # If we reach here, something went wrong
    return 0.0, {}

def should_skip_to_next_phase(phase_name, min_trials=30):
    """Determine if enough trials have been completed for a given phase"""
    try:
        study = optuna.load_study(study_name=phase_name, storage=OPTUNA_DB_URL)# type: ignore
        # Count only COMPLETE trials (not FAIL, PRUNED, etc.)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        return len(completed_trials) >= min_trials
    except Exception as e:
        logger.info(f"Error checking phase status: {e}")
        return False

def get_current_phase():
    """Determine which phase we should be working on based on trial counts"""
    # If phase 3 has trials, join phase 3
    if should_skip_to_next_phase(STUDY_PHASE3, min_trials=5):
        return 3
    # If phase 2 has enough trials, join phase 3
    elif should_skip_to_next_phase(STUDY_PHASE2, min_trials=20):
        return 3
    # If phase 1 has enough trials, join phase 2
    elif should_skip_to_next_phase(STUDY_PHASE1, min_trials=30):
        return 2
    # Otherwise start from phase 1
    else:
        return 1

def get_parameters_for_phase(phase):
    """Get best parameters from the previous phase"""
    if phase == 3:
        try:
            study2 = optuna.load_study(study_name=STUDY_PHASE2, storage=OPTUNA_DB_URL) # type: ignore
            best_phase2_params = study2.best_params
            top_params = study2.user_attrs.get("top_params", [])
            return best_phase2_params, top_params
        except Exception:
            logger.error("Could not load Phase 2 results")
            return {}, []
    
    elif phase == 2:
        try:
            study1 = optuna.load_study(study_name=STUDY_PHASE1, storage=OPTUNA_DB_URL) # type: ignore
            best_phase1_params = study1.best_params
            important_params = study1.user_attrs.get("important_params", {})
            return best_phase1_params, important_params
        except Exception:
            logger.error("Could not load Phase 1 results")
            return {}, {}
    
    return {}, {}

def main():
    """Main function with auto-phase detection and proper job status tracking"""
    # Check if OPTUNA_DB_URL is set
    if not OPTUNA_DB_URL:
        logger.error("OPTUNA_DB_URL environment variable not set or empty")
        sys.exit(1)
        
    logger.info(f"Starting hyperparameter optimization with worker {WORKER_ID}")
    logger.info(f"Using database: {OPTUNA_DB_URL}")
    
    # Create local directories
    os.makedirs("./trials", exist_ok=True)
    
    # Mark job as started
    update_job_status("started")
    
    try:
        # Determine which phase to start with
        current_phase = get_current_phase()
        logger.info(f"Based on existing trials, starting at Phase {current_phase}")
        
        # Track completion status for final reporting
        final_best_params = None
        
        # Start at the determined phase
        if current_phase == 3:
            # Get parameters from Phase 2
            best_phase2_params, top_params = get_parameters_for_phase(3)
            if best_phase2_params and len(top_params) > 0:
                logger.info(f"Continuing Phase 3 with parameters from Phase 2")
                update_job_status("running_phase3")
                phase3_value, final_best_params = run_phase3(best_phase2_params, top_params, n_trials=10)
                
                if final_best_params:
                    log_final_results(final_best_params, phase3_value)
                else:
                    logger.warning("Phase 3 did not produce valid parameters")
            else:
                update_job_status("error", error="Missing required parameters from Phase 2")
                logger.error("Cannot run Phase 3 without Phase 2 parameters")
                return
                
        elif current_phase == 2:
            # Get parameters from Phase 1
            best_phase1_params, important_params = get_parameters_for_phase(2)
            if best_phase1_params and important_params:
                logger.info(f"Starting at Phase 2 with parameters from Phase 1")
                update_job_status("running_phase2")
                phase2_value, best_phase2_params, top_params = run_phase2(
                    best_phase1_params, important_params, n_trials=20)
                
                if best_phase2_params and top_params:
                    logger.info(f"Phase 2 completed with best val_f1={phase2_value:.4f}")
                    logger.info(f"Moving to Phase 3")
                    update_job_status("running_phase3")
                    phase3_value, final_best_params = run_phase3(best_phase2_params, top_params, n_trials=15)
                    
                    if final_best_params:
                        log_final_results(final_best_params, phase3_value)
                    else:
                        logger.warning("Phase 3 did not produce valid parameters")
                else:
                    update_job_status("error", error="Phase 2 failed to produce valid parameters")
                    logger.warning("Phase 2 did not produce valid parameters, stopping here")
                    return
            else:
                update_job_status("error", error="Missing required parameters from Phase 1")
                logger.error("Cannot run Phase 2 without Phase 1 parameters")
                return
        
        else:  # Phase 1
            # Start from Phase 1
            logger.info("Starting from Phase 1")
            update_job_status("running_phase1")
            phase1_value, best_phase1_params, important_params = run_phase1(n_trials=30)
            
            if best_phase1_params and important_params:
                logger.info(f"Phase 1 completed with best val_f1={phase1_value:.4f}")
                logger.info(f"Important parameters: {list(important_params.keys())}")
                logger.info(f"Moving to Phase 2")
                update_job_status("running_phase2")
                
                phase2_value, best_phase2_params, top_params = run_phase2(
                    best_phase1_params, important_params, n_trials=30)
                
                if best_phase2_params and top_params:
                    logger.info(f"Phase 2 completed with best val_f1={phase2_value:.4f}")
                    logger.info(f"Top parameters: {top_params}")
                    logger.info(f"Moving to Phase 3")
                    update_job_status("running_phase3")
                    
                    phase3_value, final_best_params = run_phase3(
                        best_phase2_params, top_params, n_trials=15)
                    
                    if final_best_params:
                        log_final_results(final_best_params, phase3_value)
                    else:
                        logger.warning("Phase 3 did not produce valid parameters")
                else:
                    update_job_status("error", error="Phase 2 failed to produce valid parameters")
                    logger.warning("Phase 2 did not produce valid parameters, stopping here")
                    return
            else:
                update_job_status("error", error="Phase 1 failed to produce valid parameters")
                logger.warning("Phase 1 did not produce valid parameters, stopping here")
                return
        
        # Mark job as completed if we reached here
        update_job_status("completed")
        logger.info(f"Optimization completed successfully by worker {WORKER_ID}")
        
        # Create clean exit file marker for watchdog process
        try:
            with open("/tmp/optuna_clean_exit", "w") as f:
                f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logger.warning(f"Could not create clean exit marker: {e}")
            
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        update_job_status("interrupted", error="User interrupted")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error during optimization: {e}\n{error_trace}")
        update_job_status("error", error=str(e))

def cleanup_failed_trials(study_name, db_url, min_value=0.001):
    """Remove trials that failed (returned 0.0 or very low values)"""
    print(f"Cleaning up study: {study_name}")
    
    try:
        # Load the study
        study = optuna.load_study(study_name=study_name, storage=db_url)
        
        # Find failed trials
        failed_trials = [t for t in study.trials 
                        if t.state == optuna.trial.TrialState.COMPLETE 
                        and (t.value is None or t.value < min_value)]
        
        print(f"Found {len(failed_trials)} failed trials out of {len(study.trials)} total trials")
        
        if failed_trials:
            # Use the Optuna API to create a new study with filtered trials
            good_trials = [t for t in study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE 
                          and (t.value is not None and t.value >= min_value)]
            
            print(f"Keeping {len(good_trials)} good trials")
            
            # Create a new study with a temporary name
            temp_study_name = f"{study_name}_temp_{int(time.time())}"
            temp_study = optuna.create_study(
                study_name=temp_study_name,
                storage=db_url,
                direction="maximize"
            )
            
            # Copy over good trials
            for t in good_trials:
                temp_study.enqueue_trial(t.params)
            
            # Make sure the enqueued trials are processed
            if good_trials:
                temp_study.optimize(lambda _: 0.0, n_trials=len(good_trials))
            
            # Delete the old study
            optuna.delete_study(study_name=study_name, storage=db_url)
            
            # Recreate with the same name
            new_study = optuna.create_study(
                study_name=study_name,
                storage=db_url,
                direction="maximize"
            )
            
            # Copy trials from temp study
            for t in temp_study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    new_study.enqueue_trial(t.params)
                    
            # Make sure the enqueued trials are processed
            if temp_study.trials:
                new_study.optimize(lambda trial: temp_study.trials[trial.number].value, n_trials=len(temp_study.trials)) # type: ignore
            
            # Delete the temporary study
            optuna.delete_study(study_name=temp_study_name, storage=db_url)
            
            print(f"Cleanup complete. Study recreated with {len(new_study.trials)} good trials")
        else:
            print("No failed trials to clean up")
            
    except KeyError as e:
        if "Record does not exist" in str(e):
            print(f"Study {study_name} does not exist yet. Nothing to clean up.")
        else:
            raise
    except Exception as e:
        print(f"Error cleaning up study {study_name}: {e}")
        
def log_final_results(final_best_params, phase3_value):
    """Log final optimization results to MLflow and save to file"""
    logger.info(f"Phase 3 completed with best val_f1={phase3_value:.4f}")
    
    # Log to MLflow
    mlflow.set_experiment(MLFLOW_EXP_OPTUNA)
    with mlflow.start_run(run_name=f"final_best_{STUDY_BASE}"):
        mlflow.log_params(final_best_params)
        mlflow.log_metric("best_val_f1", phase3_value)
        mlflow.set_tag("study", STUDY_BASE)
        mlflow.set_tag("worker_id", WORKER_ID)
        mlflow.set_tag("completion_time", time.strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.set_tag("optimization_complete", "true")
        
        # Link to best runs from each phase
        try:
            for phase_num, study_name in [(1, STUDY_PHASE1), (2, STUDY_PHASE2), (3, STUDY_PHASE3)]:
                study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_URL) # type: ignore
                best_run_name = study.user_attrs.get('best_mlflow_run_name')
                if best_run_name:
                    mlflow.set_tag(f"best_run_phase{phase_num}", best_run_name)
        except Exception as e:
            logger.warning(f"Failed to link best phase runs: {e}")
        
    # Save to file
    try:
        with open(f'best_params_{STUDY_BASE}.json', 'w') as f:
            json.dump(final_best_params, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save best parameters: {e}")

if __name__ == "__main__":
    configure_mlflow_auth()
    cleanup_failed_trials(STUDY_PHASE1, OPTUNA_DB_URL)
    cleanup_failed_trials(STUDY_PHASE2, OPTUNA_DB_URL)
    cleanup_failed_trials(STUDY_PHASE3, OPTUNA_DB_URL)
    main()