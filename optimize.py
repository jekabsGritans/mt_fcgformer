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

# Fixed parameters
MODEL = "fcgformer"
DATASET_ID = "fb3df43da2284161aee9f043a1a4ab33"
EXPERIMENT_NAME = "fcg-hyperparam-search"
MIN_EPOCHS = 5

# Set up Optuna database URL
load_dotenv()
OPTUNA_DB_URL = os.getenv("OPTUNA_DB_URL")

# Define search strategy phases
PHASE1_STUDY_NAME = "fcgformer-phase1-exploration"
PHASE2_STUDY_NAME = "fcgformer-phase2-exploitation"
PHASE3_STUDY_NAME = "fcgformer-phase3-validation"

# Global variable to track the current process
current_process = None

# Add to optimize.py
def update_job_status(status, error=None):
    """Report job status to MLflow"""
    import mlflow
    
    try:
        # Use dedicated experiment for job monitoring
        mlflow.set_experiment("vastai_job_monitoring")
        
        with mlflow.start_run(run_name=f"optuna_{WORKER_ID}"):
            mlflow.set_tag("hostname", socket.gethostname())
            mlflow.set_tag("worker_id", WORKER_ID)
            mlflow.set_tag("status", status)
            mlflow.set_tag("last_update", time.strftime("%Y-%m-%d %H:%M:%S"))
            mlflow.set_tag("study_name", f"Current phase: {get_current_phase()}")
            
            if error:
                mlflow.set_tag("error", str(error))
            
            # Log the IP address for SSH access
            try:
                ip = subprocess.check_output("curl -s ifconfig.me", shell=True).decode().strip()
                mlflow.set_tag("ip_address", ip)
            except:
                pass
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

def suggest_parameters(trial, phase):
    """Suggest parameters based on optimization phase"""
    params = {}
    
    # === PHASE 1: Wide exploration of all parameters ===
    if phase == 1:
        # Trainer hyperparameters - wide ranges
        params["lr"] = trial.suggest_float("lr", 5e-6, 1e-3, log=True)  # Lower range
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-4, 3e-1, log=True)  # Higher min
        params["warmup_steps"] = trial.suggest_int("warmup_steps", 500, 8000, log=True)  # Longer warmup
        params["batch_size"] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        params["scheduler_t0"] = trial.suggest_int("scheduler_t0", 5, 80)
        params["scheduler_tmult"] = trial.suggest_float("scheduler_tmult", 1.0, 3.0)
        
        # Model architecture parameters - explore full range
        params["patch_size"] = trial.suggest_categorical("patch_size", [8, 16, 32])
        params["embed_dim"] = trial.suggest_int("embed_dim", 128, 896, log=True)
        params["num_layers"] = trial.suggest_int("num_layers", 1, 4)
        params["expansion_factor"] = trial.suggest_int("expansion_factor", 1, 4)
        params["n_heads"] = trial.suggest_int("n_heads", 2, 8)
        params["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)
        
        # Augmentation strategy - test different combinations
        params["use_noise"] = trial.suggest_categorical("use_noise", [True, False])
        params["use_mask"] = trial.suggest_categorical("use_mask", [True, False])
        params["use_shiftud"] = trial.suggest_categorical("use_shiftud", [True, False])
        params["use_shiftlr"] = trial.suggest_categorical("use_shiftlr", [False, True])
        params["use_revert"] = trial.suggest_categorical("use_revert", [False, True])
        
    # === PHASE 2: Exploit promising regions with narrowed search ===
    elif phase == 2:
        # Get best parameters from phase 1
        best_params = trial.study.user_attrs.get('best_phase1_params', {})
        important_params = trial.study.user_attrs.get('important_params', {})
        
        # Only tune parameters above importance threshold (or top 8)
        for param, importance in important_params.items():
            if param == "lr":
                best_lr = best_params.get("lr", 1e-4)
                params["lr"] = trial.suggest_float("lr", best_lr * 0.3, best_lr * 3.0, log=True)
            
            elif param == "batch_size":
                # Keep categorical but maybe prioritize values near best
                params["batch_size"] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
            
            elif param == "weight_decay":
                best_wd = best_params.get("weight_decay", 0.01)
                params["weight_decay"] = trial.suggest_float("weight_decay", best_wd * 0.3, best_wd * 3.0, log=True)
                
            elif param == "warmup_steps":
                best_ws = best_params.get("warmup_steps", 1000)
                params["warmup_steps"] = trial.suggest_int("warmup_steps", int(best_ws * 0.5), int(best_ws * 2.0), log=True)
                
            elif param == "scheduler_t0":
                best_t0 = best_params.get("scheduler_t0", 40)
                params["scheduler_t0"] = trial.suggest_int("scheduler_t0", max(5, int(best_t0 * 0.5)), int(best_t0 * 1.5))
                
            elif param == "scheduler_tmult":
                best_tmult = best_params.get("scheduler_tmult", 2.0)
                params["scheduler_tmult"] = trial.suggest_float("scheduler_tmult", max(1.0, best_tmult * 0.7), best_tmult * 1.3)
                
            elif param == "patch_size":
                # Keep categorical
                params["patch_size"] = trial.suggest_categorical("patch_size", [8, 16, 32])
                
            elif param == "embed_dim":
                best_embed = best_params.get("embed_dim", 512)
                params["embed_dim"] = trial.suggest_int("embed_dim", int(best_embed * 0.7), int(best_embed * 1.3), log=True)
                
            elif param == "num_layers":
                best_layers = best_params.get("num_layers", 2)
                params["num_layers"] = trial.suggest_int("num_layers", max(1, best_layers - 1), best_layers + 1)
                
            elif param == "expansion_factor":
                best_exp = best_params.get("expansion_factor", 2)
                params["expansion_factor"] = trial.suggest_int("expansion_factor", max(1, best_exp - 1), best_exp + 1)
                
            elif param == "n_heads":
                best_heads = best_params.get("n_heads", 4)
                params["n_heads"] = trial.suggest_int("n_heads", max(2, best_heads - 2), best_heads + 2)
                
            elif param == "dropout_p":
                best_drop = best_params.get("dropout_p", 0.1)
                params["dropout_p"] = trial.suggest_float("dropout_p", max(0.0, best_drop - 0.1), min(0.5, best_drop + 0.1))
                
        # For augmentation parameters, keep the best configuration from phase 1
        for aug_param in ["use_noise", "use_mask", "use_shiftud", "use_shiftlr", "use_revert"]:
            params[aug_param] = best_params.get(aug_param, True)
        
        # Fill in parameters not in important_params with best values from phase 1
        for param_name, param_value in best_params.items():
            if param_name not in params:
                params[param_name] = param_value
    
    # === PHASE 3: Validation with focused parameter tuning ===
    elif phase == 3:
        # Get best parameters from phase 2
        best_params = trial.study.user_attrs.get('best_phase2_params', {})
        top_params = trial.study.user_attrs.get('top_params', {})
        
        # Only tune the top 3-5 most important parameters
        for param in top_params:
            if param == "lr":
                best_lr = best_params.get("lr", 1e-4)
                params["lr"] = trial.suggest_float("lr", best_lr * 0.7, best_lr * 1.3, log=True)
            
            elif param == "batch_size":
                # In validation phase, we might want to fix batch size or try very few options
                params["batch_size"] = trial.suggest_categorical('batch_size', [best_params.get("batch_size", 128)])
            
            elif param == "weight_decay":
                best_wd = best_params.get("weight_decay", 0.01)
                params["weight_decay"] = trial.suggest_float("weight_decay", best_wd * 0.7, best_wd * 1.3, log=True)
                
            elif param == "warmup_steps":
                best_ws = best_params.get("warmup_steps", 1000)
                params["warmup_steps"] = trial.suggest_int("warmup_steps", int(best_ws * 0.8), int(best_ws * 1.2), log=True)
                
            elif param == "scheduler_t0":
                best_t0 = best_params.get("scheduler_t0", 40)
                params["scheduler_t0"] = trial.suggest_int("scheduler_t0", max(5, int(best_t0 * 0.8)), int(best_t0 * 1.2))
                
            elif param == "scheduler_tmult":
                best_tmult = best_params.get("scheduler_tmult", 2.0)
                params["scheduler_tmult"] = trial.suggest_float("scheduler_tmult", max(1.0, best_tmult * 0.9), best_tmult * 1.1)
                
            elif param == "patch_size":
                # Usually fixed in phase 3
                params["patch_size"] = best_params.get("patch_size", 16)
                
            elif param == "embed_dim":
                best_embed = best_params.get("embed_dim", 512)
                params["embed_dim"] = trial.suggest_int("embed_dim", int(best_embed * 0.9), int(best_embed * 1.1), log=True)
                
            elif param == "num_layers":
                # Usually fixed in phase 3
                params["num_layers"] = best_params.get("num_layers", 2)
                
            elif param == "expansion_factor":
                # Usually fixed in phase 3
                params["expansion_factor"] = best_params.get("expansion_factor", 2)
                
            elif param == "n_heads":
                # Usually fixed in phase 3
                params["n_heads"] = best_params.get("n_heads", 4)
                
            elif param == "dropout_p":
                best_drop = best_params.get("dropout_p", 0.1)
                params["dropout_p"] = trial.suggest_float("dropout_p", max(0.0, best_drop - 0.05), min(0.5, best_drop + 0.05))
            
            # Handle augmentation parameters
            elif param == "use_noise" or param == "noise_prob" or param == "noise_snr_min" or param == "noise_snr_max":
                # Keep augmentation settings from phase 2, or refine if specifically identified
                params["use_noise"] = best_params.get("use_noise", True)
                if param == "noise_prob":
                    best_prob = best_params.get("noise_prob", 0.3)
                    params["noise_prob"] = trial.suggest_float("noise_prob", best_prob * 0.8, min(0.5, best_prob * 1.2))
            
            elif param == "use_mask" or param == "mask_prob" or param == "mask_min" or param == "mask_max":
                params["use_mask"] = best_params.get("use_mask", True)
                if param == "mask_prob":
                    best_prob = best_params.get("mask_prob", 0.3)
                    params["mask_prob"] = trial.suggest_float("mask_prob", best_prob * 0.8, min(0.5, best_prob * 1.2))
            
            elif param == "use_shiftud" or param == "shiftud_prob":
                params["use_shiftud"] = best_params.get("use_shiftud", True)
                if param == "shiftud_prob":
                    best_prob = best_params.get("shiftud_prob", 0.3)
                    params["shiftud_prob"] = trial.suggest_float("shiftud_prob", best_prob * 0.8, min(0.5, best_prob * 1.2))
        
        # Set all other parameters to their best values from phase 2
        for param_name, param_value in best_params.items():
            if param_name not in params:
                params[param_name] = param_value
    
    return params

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    global current_process

    # Clean up old trials if too many exist
    trial_dirs = glob.glob("./trials/phase*_trial_*")
    if len(trial_dirs) > 20:  # Keep only most recent 50 trials
        for old_dir in sorted(trial_dirs, key=os.path.getctime)[:len(trial_dirs)-50]:
            try:
                shutil.rmtree(old_dir)
            except:
                pass  # Ignore errors

    # Get current phase from study user attrs
    phase = trial.study.user_attrs.get('phase', 1)
    
    # Get parameters for current phase
    params = suggest_parameters(trial, phase)
    
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
    
    # Create a unique directory for this trial's files
    trial_metrics_dir = os.path.join("./trials", f"phase{phase}_trial_{trial.number}")
    os.makedirs(trial_metrics_dir, exist_ok=True)
    metrics_file = os.path.join(trial_metrics_dir, "metrics.json")
    training_log = os.path.join(trial_metrics_dir, "training.log")
    
    # Build command with base parameters
    cmd = [
        "python", "main.py",
        "mode=train",
        "device=cuda:0",
        f"experiment_name={EXPERIMENT_NAME}_phase{phase}",
        f"dataset_id={DATASET_ID}",
        f"model={MODEL}",
        
        # Trainer params
        f"trainer.lr={lr}",
        f"trainer.batch_size={batch_size}",
        f"trainer.weight_decay={weight_decay}",
        f"trainer.warmup_steps={warmup_steps}",
        f"trainer.scheduler_t0={scheduler_t0}",
        f"trainer.scheduler_tmult={scheduler_tmult}",
        
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
        f"trainer.epochs={300 if phase == 3 else 200 if phase == 2 else 100}",
        f"trainer.patience={200 if phase == 3 else 100 if phase == 2 else 50}",
        f"metric_output_file={metrics_file}",
        
        # Dataset weights
        "nist_weight=1.0",
        "nist_lser_weight=0.0",
        "chemmotion_weight=0.0",
        "chemmotion_lser_weight=0.0",
        "graphformer_weight=0.0",
        "graphformer_lser_weight=0.0",
    ]
    
    # Add transform-specific parameters when enabled
    if use_noise:
        noise_prob = trial.suggest_float("noise_prob", 0.1, 0.5) if phase == 1 else params.get("noise_prob", 0.3)
        noise_snr_min = trial.suggest_int("noise_snr_min", 1, 5) if phase == 1 else params.get("noise_snr_min", 2)
        noise_snr_max = trial.suggest_int("noise_snr_max", 10, 30) if phase == 1 else params.get("noise_snr_max", 20)
        cmd.extend([
            f"noise_prob={noise_prob}",
            f"noise_snr_min={noise_snr_min}",
            f"noise_snr_max={noise_snr_max}",
        ])
    
    if use_mask:
        mask_prob = trial.suggest_float("mask_prob", 0.1, 0.5) if phase == 1 else params.get("mask_prob", 0.3)
        mask_min = trial.suggest_float("mask_min", 0.01, 0.1) if phase == 1 else params.get("mask_min", 0.01)
        mask_max = trial.suggest_float("mask_max", 0.1, 0.3) if phase == 1 else params.get("mask_max", 0.15)
        cmd.extend([
            f"mask_prob={mask_prob}",
            f"mask_min={mask_min}",
            f"mask_max={mask_max}",
        ])
    
    if use_shiftud:
        shiftud_prob = trial.suggest_float("shiftud_prob", 0.1, 0.5) if phase == 1 else params.get("shiftud_prob", 0.3)
        shiftud_min = trial.suggest_float("shiftud_min", 0.01, 0.1) if phase == 1 else params.get("shiftud_min", 0.01)
        shiftud_max = trial.suggest_float("shiftud_max", 0.1, 0.2) if phase == 1 else params.get("shiftud_max", 0.1)
        cmd.extend([
            f"shiftud_prob={shiftud_prob}",
            f"shiftud_min={shiftud_min}",
            f"shiftud_max={shiftud_max}",
        ])
    
    if use_shiftlr:
        shiftlr_prob = trial.suggest_float("shiftlr_prob", 0.1, 0.5) if phase == 1 else params.get("shiftlr_prob", 0.3)
        shiftlr_min = trial.suggest_float("shiftlr_min", 0.01, 0.1) if phase == 1 else params.get("shiftlr_min", 0.01)
        shiftlr_max = trial.suggest_float("shiftlr_max", 0.1, 0.2) if phase == 1 else params.get("shiftlr_max", 0.1)
        cmd.extend([
            f"shiftlr_prob={shiftlr_prob}",
            f"shiftlr_min={shiftlr_min}",
            f"shiftlr_max={shiftlr_max}",
        ])
    
    if use_revert:
        revert_prob = trial.suggest_float("revert_prob", 0.1, 0.5) if phase == 1 else params.get("revert_prob", 0.3)
        cmd.append(f"revert_prob={revert_prob}")
    
    # Log detailed trial information
    transform_info = f"Transforms: noise={use_noise}, mask={use_mask}, shiftUD={use_shiftud}, shiftLR={use_shiftlr}, revert={use_revert}"
    model_info = f"Model: layers={num_layers}, heads={n_heads}, embed_dim={embed_dim}, patch={patch_size}"
    phase_info = f"Phase {phase}"
    logger.info(f"Starting trial {trial.number} ({phase_info}) with {model_info}, {transform_info}")
    
    try:
        # Start the training process with output redirected to log file
        with open(training_log, 'w') as log_file:
            current_process = subprocess.Popen(
                cmd, 
                stdout=log_file, 
                stderr=subprocess.STDOUT
            )
        
        # Monitoring code with phase-specific adjustments
        best_val_f1 = 0
        last_epoch = 0
        no_improvement_count = 0
        
        # Adjust patience and early stopping based on phase
        early_stop_epochs = 60 if phase == 1 else 100 if phase == 2 else 150
        patience_threshold = 15 if phase == 1 else 20 if phase == 2 else 30
        min_f1_threshold = 0.3 if phase == 1 else 0.4 if phase == 2 else 0.5

        
        while current_process.poll() is None:
            time.sleep(5)  # Check metrics every 5 seconds
            
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
                                
                except (json.JSONDecodeError, IOError) as e:
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
        return 0.0
        
    except Exception as e:
        logger.error(f"Error in trial {trial.number} P{phase}: {e}")
        if current_process and current_process.poll() is None:
            gracefully_terminate_process(current_process)
        return 0.0
        
    finally:
        current_process = None

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

def run_phase1(n_trials=40):
    """Phase 1: Wide exploration with per-trial phase advancement check"""
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    logger.info(f"Phase 1: Starting EXPLORATION with up to {n_trials} trials")
    
    # Create phase 1 study with RandomSampler for better exploration
    study = optuna.create_study(
        study_name=PHASE1_STUDY_NAME,
        storage=OPTUNA_DB_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8, 
            n_warmup_steps=15,
            interval_steps=2
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
            
            return study.best_value, best_params, important_params
            
    except KeyboardInterrupt:
        logger.info("Phase 1 interrupted by user")
    except Exception as e:
        logger.error(f"Error during Phase 1: {e}")
        
    # If we reach here, something went wrong
    return 0.0, {}, {}

def run_phase2(best_phase1_params, important_params, n_trials=30):
    """Phase 2: Exploitation with per-trial phase advancement check"""
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    logger.info(f"Phase 2: Starting EXPLOITATION with up to {n_trials} trials")
    logger.info(f"Using top parameters: {list(important_params.keys())}")
    
    # Create phase 2 study with TPESampler for exploitation
    study = optuna.create_study(
        study_name=PHASE2_STUDY_NAME,
        storage=OPTUNA_DB_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8, 
            n_warmup_steps=15,
            interval_steps=2
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
            
            return study.best_value, best_params, top_params
            
    except KeyboardInterrupt:
        logger.info("Phase 2 interrupted by user")
    except Exception as e:
        logger.error(f"Error during Phase 2: {e}")
        
    # If we reach here, something went wrong
    return 0.0, {}, []


def run_phase3(best_phase2_params, top_params, n_trials=15):
    """Phase 3: Final validation with per-trial phase advancement check"""
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    logger.info(f"Phase 3: Starting VALIDATION with up to {n_trials} trials")
    logger.info(f"Fine-tuning parameters: {top_params}")
    
    # Create phase 3 study with focused TPESampler
    study = optuna.create_study(
        study_name=PHASE3_STUDY_NAME,
        storage=OPTUNA_DB_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8, 
            n_warmup_steps=15,
            interval_steps=2
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
        study = optuna.load_study(study_name=phase_name, storage=OPTUNA_DB_URL)
        # Count only COMPLETE trials (not FAIL, PRUNED, etc.)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        return len(completed_trials) >= min_trials
    except Exception as e:
        logger.info(f"Error checking phase status: {e}")
        return False

def get_current_phase():
    """Determine which phase we should be working on based on trial counts"""
    # If phase 3 has trials, join phase 3
    if should_skip_to_next_phase(PHASE3_STUDY_NAME, min_trials=5):
        return 3
    # If phase 2 has enough trials, join phase 3
    elif should_skip_to_next_phase(PHASE2_STUDY_NAME, min_trials=30):
        return 3
    # If phase 1 has enough trials, join phase 2
    elif should_skip_to_next_phase(PHASE1_STUDY_NAME, min_trials=40):
        return 2
    # Otherwise start from phase 1
    else:
        return 1

def get_parameters_for_phase(phase):
    """Get best parameters from the previous phase"""
    if phase == 3:
        try:
            study2 = optuna.load_study(study_name=PHASE2_STUDY_NAME, storage=OPTUNA_DB_URL)
            best_phase2_params = study2.best_params
            top_params = study2.user_attrs.get("top_params", [])
            return best_phase2_params, top_params
        except Exception:
            logger.error("Could not load Phase 2 results")
            return {}, []
    
    elif phase == 2:
        try:
            study1 = optuna.load_study(study_name=PHASE1_STUDY_NAME, storage=OPTUNA_DB_URL)
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
                phase3_value, final_best_params = run_phase3(best_phase2_params, top_params, n_trials=15)
                
                if final_best_params:
                    logger.info(f"Phase 3 completed with best val_f1={phase3_value:.4f}")
                    # Save final best parameters
                    try:
                        with open(f'best_params_{WORKER_ID}.json', 'w') as f:
                            json.dump(final_best_params, f, indent=2)
                    except Exception as e:
                        logger.error(f"Failed to save best parameters: {e}")
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
                    best_phase1_params, important_params, n_trials=30)
                
                if best_phase2_params and top_params:
                    logger.info(f"Phase 2 completed with best val_f1={phase2_value:.4f}")
                    logger.info(f"Moving to Phase 3")
                    update_job_status("running_phase3")
                    phase3_value, final_best_params = run_phase3(best_phase2_params, top_params, n_trials=15)
                    
                    if final_best_params:
                        logger.info(f"Phase 3 completed with best val_f1={phase3_value:.4f}")
                        # Save final best parameters
                        try:
                            with open(f'best_params_{WORKER_ID}.json', 'w') as f:
                                json.dump(final_best_params, f, indent=2)
                        except Exception as e:
                            logger.error(f"Failed to save best parameters: {e}")
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
            phase1_value, best_phase1_params, important_params = run_phase1(n_trials=40)
            
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
                        logger.info(f"Phase 3 completed with best val_f1={phase3_value:.4f}")
                        # Save final best parameters
                        try:
                            with open(f'best_params_{WORKER_ID}.json', 'w') as f:
                                json.dump(final_best_params, f, indent=2)
                        except Exception as e:
                            logger.error(f"Failed to save best parameters: {e}")
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
        
if __name__ == "__main__":
    main()