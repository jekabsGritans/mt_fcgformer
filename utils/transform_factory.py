import utils.transforms as T


def create_eval_transform():
    """Creates a fixed evaluation transform pipeline"""
    # Hard-coded signal size for all transforms
    signal_size = 1024
    
    # Always-used transforms
    normalizer = T.Normalizer(with_std=False)
    resizer = T.Resizer(signal_size=signal_size)
    
    # Compose the evaluation transforms
    eval_transforms = T.Compose([normalizer, resizer])
    return eval_transforms

def create_transforms(cfg):
    """Creates transform pipelines based on simplified config parameters"""
    # Hard-coded signal size for all transforms
    signal_size = 1024
    
    # Always-used transforms
    normalizer = T.Normalizer(with_std=False)
    resizer = T.Resizer(signal_size=signal_size)
    
    # Eval transforms are fixed
    eval_transforms = T.Compose([normalizer, resizer])
    
    # Training transforms depend on config
    train_transforms = []
    
    # Add optional noise augmentation
    if cfg.get("use_noise", True):
        noise_prob = cfg.get("noise_prob", 0.3)
        noise_snr_min = cfg.get("noise_snr_min", 2)
        noise_snr_max = cfg.get("noise_snr_max", 20)
        train_transforms.append(
            T.AddNoise(prob=noise_prob, snr_range=[noise_snr_min, noise_snr_max]) # type: ignore
        )
    
    # Add optional revert augmentation
    if cfg.get("use_revert", False):
        revert_prob = cfg.get("revert_prob", 0.3)
        train_transforms.append(T.Revert(prob=revert_prob))
    
    # Add optional masking augmentation
    if cfg.get("use_mask", True):
        mask_prob = cfg.get("mask_prob", 0.3)
        mask_min = cfg.get("mask_min", 0.01)
        mask_max = cfg.get("mask_max", 0.15)
        train_transforms.append(
            T.MaskZeros(prob=mask_prob, mask_p=[mask_min, mask_max]) # type: ignore
        )
    
    # Add optional horizontal shift augmentation
    if cfg.get("use_shiftlr", False):
        shiftlr_prob = cfg.get("shiftlr_prob", 0.3)
        shiftlr_min = cfg.get("shiftlr_min", 0.01)
        shiftlr_max = cfg.get("shiftlr_max", 0.1)
        train_transforms.append(
            T.ShiftLR(prob=shiftlr_prob, shift_p=[shiftlr_min, shiftlr_max]) # type: ignore
        )
    
    # Add optional vertical shift augmentation
    if cfg.get("use_shiftud", True):
        shiftud_prob = cfg.get("shiftud_prob", 0.3)
        shiftud_min = cfg.get("shiftud_min", 0.01)
        shiftud_max = cfg.get("shiftud_max", 0.1)
        train_transforms.append(
            T.ShiftUD(prob=shiftud_prob, shift_p=[shiftud_min, shiftud_max]) # type: ignore
        )
    
    # Always add normalizer and resizer at the end
    train_transforms.extend([normalizer, resizer])
    
    return T.Compose(train_transforms), eval_transforms