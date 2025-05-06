## TODO
- [x] finish basic train/eval/predict with proper configs and scripts
- [x] test modes of running and debug
- [ ] refactor / simplify (global config, singleton MLFlowManager)
- [ ] better logging. and save log to mlflow
- [ ] proper trainer (LR scheduling, etc.) and evaluation (per-class accuracies, visualizations, etc.)
- [ ] implement fcg transformer
- [ ] dataset registry in mlfow
- [ ] deployable mlflow models

## Datasets
### FTIR
- to use the FTIR dataset used in the FCG-Former paper:
    - download `dataset.zip` from https://huggingface.co/datasets/lycaoduong/FTIR/tree/main 
    - unzip it in `data/ftir`