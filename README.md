## TODO
- [x] finish basic train/eval/predict with proper configs and scripts
- [x] test modes of running and debug
- [x] refactor / simplify (global config, singleton MLFlowManager)
- [x] better logging. and save log to mlflow
- [ ] proper trainer (LR scheduling, etc.) and evaluation (per-class accuracies, visualizations, etc.)
    - rethin pos_weights given multilabel
- [ ] implement fcg transformer
- [ ] dataset registry in mlfow
- [ ] deployable mlflow models

## For model improvements 
- [ ] templates for reproducible dataset generation through augmentation of existing datasets in notebooks 
    - 2 kinds of transforms:
        - simple augmentations like adding noise, done at training time
        - theory-based augmentations to expand dataset at time of versioned dataset generation

## Datasets
### FTIR
- to use the FTIR dataset used in the FCG-Former paper:
    - download `dataset.zip` from https://huggingface.co/datasets/lycaoduong/FTIR/tree/main 
    - unzip it in `data/ftir`