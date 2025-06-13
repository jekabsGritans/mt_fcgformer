
def suggest_parameters(trial, phase):
    """Suggest parameters based on optimization phase"""
    if phase == 1:
        return suggest_parameters_phase1(trial)
    elif phase == 2:
        return suggest_parameters_phase2(trial)
    elif phase == 3:
        return suggest_parameters_phase3(trial)
    else:
        raise ValueError(f"Unknown phase: {phase}")

def suggest_parameters_phase1(trial):
    """Phase 1: Wide exploration of all parameters"""
    params = {}
    
    # Trainer hyperparameters - wide ranges
    params["lr"] = trial.suggest_float("lr", 5e-6, 1e-3, log=True)  # Lower range
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-4, 3e-1, log=True)
    params["warmup_steps"] = trial.suggest_int("warmup_steps", 500, 2000)  
    params["batch_size"] = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    params["scheduler_t0"] = trial.suggest_int("scheduler_t0", 1, 10)
    params["scheduler_tmult"] = trial.suggest_categorical("scheduler_tmult", [1, 2])

    # Auxiliary loss parameters
    params["initial_aux_bool_weight"] = trial.suggest_float("initial_aux_bool_weight", 0.01, 1.0, log=True)
    params["initial_aux_float_weight"] = trial.suggest_float("initial_aux_float_weight", 0.0001, 0.01, log=True)
    params["aux_epochs"] = trial.suggest_int("aux_epochs", 10, 40)
    
    # Dataset weights (nist_weight always fixed at 1.0 as baseline)
    params["nist_lser_weight"] = trial.suggest_float("nist_lser_weight", 0.0, 0.5)
    params["chemmotion_weight"] = trial.suggest_float("chemmotion_weight", 0.0, 1.0)
    params["chemmotion_lser_weight"] = trial.suggest_float("chemmotion_lser_weight", 0.0, 0.3)
    params["graphformer_weight"] = trial.suggest_float("graphformer_weight", 0.0, 0.5)
    params["graphformer_lser_weight"] = trial.suggest_float("graphformer_lser_weight", 0.0, 0.5)
    
    # Model architecture parameters - explore full range
    params["patch_size"] = trial.suggest_categorical("patch_size", [8, 16, 32])
    params["embed_dim"] = trial.suggest_int("embed_dim", 128, 896, step=8)
    params["num_layers"] = trial.suggest_int("num_layers", 1, 4)
    params["expansion_factor"] = trial.suggest_int("expansion_factor", 1, 4)
    params["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 8])
    params["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)
    
    # Augmentation strategy - test different combinations
    params["use_noise"] = trial.suggest_categorical("use_noise", [True, False])
    params["use_mask"] = trial.suggest_categorical("use_mask", [True, False])
    params["use_shiftud"] = trial.suggest_categorical("use_shiftud", [True, False])
    params["use_shiftlr"] = trial.suggest_categorical("use_shiftlr", [False, True])
    params["use_revert"] = trial.suggest_categorical("use_revert", [False, True])
    
    return params

def suggest_parameters_phase2(trial):
    """Phase 2: Exploit promising regions with narrowed search"""
    params = {}
    
    # Get best parameters from phase 1
    best_params = trial.study.user_attrs.get('best_phase1_params', {})
    important_params = trial.study.user_attrs.get('important_params', {})
    
    # Only tune parameters above importance threshold (or top 8)
    for param, importance in important_params.items():
        params = suggest_param_phase2(trial, param, best_params, params)
    
    # For parameters not explicitly chosen, fill with best values from phase 1
    aux_loss_params = ["initial_aux_bool_weight", "initial_aux_float_weight", "aux_epochs"]
    dataset_weight_params = ["nist_lser_weight", "chemmotion_weight", "chemmotion_lser_weight", 
                            "graphformer_weight", "graphformer_lser_weight"]
    
    for param in aux_loss_params + dataset_weight_params:
        if param not in params and param in best_params:
            params[param] = best_params[param]
            
    # For augmentation parameters, keep the best configuration from phase 1
    for aug_param in ["use_noise", "use_mask", "use_shiftud", "use_shiftlr", "use_revert"]:
        params[aug_param] = best_params.get(aug_param)
    
    # Fill in parameters not in important_params with best values from phase 1
    for param_name, param_value in best_params.items():
        if param_name not in params:
            params[param_name] = param_value
            
    return params

def suggest_param_phase2(trial, param, best_params, params):
    """Suggest a specific parameter for phase 2"""
    if param == "lr":
        best_lr = best_params.get("lr", 1e-4)
        params["lr"] = trial.suggest_float("lr", best_lr * 0.3, best_lr * 3.0, log=True)
    
    elif param == "batch_size":
        # Keep categorical but maybe prioritize values near best
        params["batch_size"] = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    
    elif param == "weight_decay":
        best_wd = best_params.get("weight_decay")
        params["weight_decay"] = trial.suggest_float("weight_decay", best_wd * 0.3, best_wd * 3.0, log=True)
        
    elif param == "warmup_steps":
        best_ws = best_params.get("warmup_steps")
        params["warmup_steps"] = trial.suggest_int("warmup_steps", int(best_ws * 0.5), int(best_ws * 2.0))
        
    elif param == "scheduler_t0":
        best_t0 = best_params.get("scheduler_t0")
        params["scheduler_t0"] = trial.suggest_int("scheduler_t0", max(1, int(best_t0 * 0.5)), int(best_t0 * 1.5))
        
    elif param == "scheduler_tmult":
        best_tmult = best_params.get("scheduler_tmult")
        params["scheduler_tmult"] = best_tmult
        
    elif param == "patch_size":
        params["patch_size"] = trial.suggest_categorical("patch_size", [8, 16, 32])
        
    elif param == "embed_dim":
        best_embed = best_params.get("embed_dim")
        min_rounded = int(best_embed * 0.7 // 8) * 8
        max_rounded = int(best_embed * 1.3 // 8) * 8
        params["embed_dim"] = trial.suggest_int("embed_dim", min_rounded, max_rounded, step=8)
        
    elif param == "num_layers":
        best_layers = best_params.get("num_layers")
        params["num_layers"] = trial.suggest_int("num_layers", max(1, best_layers - 1), best_layers + 1)
        
    elif param == "expansion_factor":
        best_exp = best_params.get("expansion_factor")
        params["expansion_factor"] = trial.suggest_int("expansion_factor", max(1, best_exp - 1), best_exp + 1)
        
    elif param == "n_heads":
        params["n_heads"] = best_params.get("n_heads")
        
    elif param == "dropout_p":
        best_drop = best_params.get("dropout_p")
        params["dropout_p"] = trial.suggest_float("dropout_p", max(0.0, best_drop - 0.1), min(0.5, best_drop + 0.1))

    # Auxiliary loss parameters
    elif param == "initial_aux_bool_weight":
        best_weight = best_params.get("initial_aux_bool_weight")
        params["initial_aux_bool_weight"] = trial.suggest_float(
            "initial_aux_bool_weight", 
            max(0.05, best_weight * 0.5), 
            min(1.0, best_weight * 1.5)
        )
        
    elif param == "initial_aux_float_weight":
        best_weight = best_params.get("initial_aux_float_weight")
        params["initial_aux_float_weight"] = trial.suggest_float(
            "initial_aux_float_weight", 
            best_weight * 0.5, 
            best_weight * 2.0, 
            log=True
        )
        
    elif param == "aux_epochs":
        best_epochs = best_params.get("aux_epochs")
        params["aux_epochs"] = trial.suggest_int(
            "aux_epochs", 
            max(20, int(best_epochs * 0.7)), 
            int(best_epochs * 1.3)
        )
        
    # Dataset weights
    elif param == "nist_lser_weight":
        best_weight = best_params.get("nist_lser_weight")
        params["nist_lser_weight"] = trial.suggest_float(
            "nist_lser_weight", 
            max(0.0, best_weight - 0.1), 
            min(0.6, best_weight + 0.1)
        )
        
    elif param == "chemmotion_weight":
        best_weight = best_params.get("chemmotion_weight")
        params["chemmotion_weight"] = trial.suggest_float(
            "chemmotion_weight", 
            max(0.0, best_weight - 0.2), 
            min(1.0, best_weight + 0.2)
        )
        
    elif param == "chemmotion_lser_weight":
        best_weight = best_params.get("chemmotion_lser_weight")
        params["chemmotion_lser_weight"] = trial.suggest_float(
            "chemmotion_lser_weight", 
            max(0.0, best_weight - 0.1), 
            min(0.4, best_weight + 0.1)
        )
        
    elif param == "graphformer_weight":
        best_weight = best_params.get("graphformer_weight")
        params["graphformer_weight"] = trial.suggest_float(
            "graphformer_weight", 
            max(0.0, best_weight - 0.1), 
            min(0.6, best_weight + 0.1)
        )
        
    elif param == "graphformer_lser_weight":
        best_weight = best_params.get("graphformer_lser_weight")
        params["graphformer_lser_weight"] = trial.suggest_float(
            "graphformer_lser_weight", 
            max(0.0, best_weight - 0.1), 
            min(0.6, best_weight + 0.1)
        )
        
    return params

def suggest_parameters_phase3(trial):
    """Phase 3: Validation with focused parameter tuning"""
    params = {}
    
    # Get best parameters from phase 2
    best_params = trial.study.user_attrs.get('best_phase2_params')
    top_params = trial.study.user_attrs.get('top_params')
    
    # Only tune the top 3-5 most important parameters
    for param in top_params:
        params = suggest_param_phase3(trial, param, best_params, params)
    
    # Set all augmentation parameters to their best values
    for aug_param in ["use_noise", "use_mask", "use_shiftud", "use_shiftlr", "use_revert"]:
        params[aug_param] = best_params.get(aug_param, False)
    
    # Set all other parameters to their best values from phase 2
    for param_name, param_value in best_params.items():
        if param_name not in params:
            params[param_name] = param_value
    
    return params

def suggest_param_phase3(trial, param, best_params, params):
    """Suggest a specific parameter for phase 3"""
    if param == "lr":
        best_lr = best_params.get("lr")
        params["lr"] = trial.suggest_float("lr", best_lr * 0.8, best_lr * 1.2, log=True)
    
    elif param == "batch_size":
        # In validation phase, we might want to fix batch size or try very few options
        params["batch_size"] = trial.suggest_categorical('batch_size', [best_params.get("batch_size", 128)])
    
    elif param == "weight_decay":
        best_wd = best_params.get("weight_decay")
        params["weight_decay"] = trial.suggest_float("weight_decay", best_wd * 0.7, best_wd * 1.3, log=True)
        
    elif param == "warmup_steps":
        best_ws = best_params.get("warmup_steps")
        params["warmup_steps"] = trial.suggest_int("warmup_steps", int(best_ws * 0.8), int(best_ws * 1.2))
        
    elif param == "scheduler_t0":
        best_t0 = best_params.get("scheduler_t0")
        params["scheduler_t0"] = trial.suggest_int("scheduler_t0", max(1, int(best_t0 * 0.8)), int(best_t0 * 1.2))
        
    elif param == "scheduler_tmult":
        best_tmult = best_params.get("scheduler_tmult")
        params["scheduler_tmult"] = best_tmult
        
    elif param == "patch_size":
        # Usually fixed in phase 3
        params["patch_size"] = best_params.get("patch_size")
        
    elif param == "embed_dim":
        best_embed = best_params.get("embed_dim")
        min_rounded = int(best_embed * 0.9 // 8) * 8
        max_rounded = int(best_embed * 1.1 // 8) * 8
        params["embed_dim"] = trial.suggest_int("embed_dim", min_rounded, max_rounded, step=8)
        
    elif param == "num_layers":
        params["num_layers"] = best_params.get("num_layers")
        
    elif param == "expansion_factor":
        params["expansion_factor"] = best_params.get("expansion_factor")
        
    elif param == "n_heads":
        params["n_heads"] = best_params.get("n_heads")
        
    elif param == "dropout_p":
        best_drop = best_params.get("dropout_p")
        params["dropout_p"] = trial.suggest_float("dropout_p", max(0.0, best_drop - 0.05), min(0.5, best_drop + 0.05))

    elif param == "initial_aux_bool_weight":
        best_weight = best_params.get("initial_aux_bool_weight")
        params["initial_aux_bool_weight"] = trial.suggest_float(
            "initial_aux_bool_weight", 
            max(0.05, best_weight * 0.8), 
            min(1.0, best_weight * 1.2)
        )
    
    elif param == "initial_aux_float_weight":
        best_weight = best_params.get("initial_aux_float_weight")
        params["initial_aux_float_weight"] = trial.suggest_float(
            "initial_aux_float_weight", 
            best_weight * 0.7, 
            best_weight * 1.3, 
            log=True
        )

    elif param == "aux_epochs":
        best_epochs = best_params.get("aux_epochs")
        params["aux_epochs"] = trial.suggest_int(
            "aux_epochs", 
            max(20, int(best_epochs * 0.9)), 
            int(best_epochs * 1.1)
        )
        
    # Dataset weights
    elif param == "nist_lser_weight":
        best_weight = best_params.get("nist_lser_weight")
        params["nist_lser_weight"] = trial.suggest_float(
            "nist_lser_weight", 
            max(0.0, best_weight - 0.05), 
            min(0.6, best_weight + 0.05)
        )
        
    elif param == "chemmotion_weight":
        best_weight = best_params.get("chemmotion_weight")
        params["chemmotion_weight"] = trial.suggest_float(
            "chemmotion_weight", 
            max(0.0, best_weight - 0.1), 
            min(1.0, best_weight + 0.1)
        )
        
    elif param == "chemmotion_lser_weight":
        best_weight = best_params.get("chemmotion_lser_weight")
        params["chemmotion_lser_weight"] = trial.suggest_float(
            "chemmotion_lser_weight", 
            max(0.0, best_weight - 0.05), 
            min(0.4, best_weight + 0.05)
        )
        
    elif param == "graphformer_weight":
        best_weight = best_params.get("graphformer_weight")
        params["graphformer_weight"] = trial.suggest_float(
            "graphformer_weight", 
            max(0.0, best_weight - 0.05), 
            min(0.6, best_weight + 0.05)
        )
    
    elif param == "graphformer_lser_weight":
        best_weight = best_params.get("graphformer_lser_weight")
        params["graphformer_lser_weight"] = trial.suggest_float(
            "graphformer_lser_weight", 
            max(0.0, best_weight - 0.05), 
            min(0.6, best_weight + 0.05)
        )
    
    return params