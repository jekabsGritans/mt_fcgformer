## TODO
- [x] finish basic train/eval/predict with proper configs and scripts
- [ ] test modes of running and debug
- [ ] integrate mlflow and more metrics (from paper) / monitoring (use logger as well) / checkpointing
- [ ] proper trainer (LR scheduling, etc.) and evaluation (per-class accuracies, etc.)
- [ ] dockerize (first custom deployable, then myb dev container)
- [ ] dataset registry in mlfow
- [ ] implement fcg transformer

## Datasets
### FTIR
- to use the FTIR dataset used in the FCG-Former paper:
    - download `dataset.zip` from https://huggingface.co/datasets/lycaoduong/FTIR/tree/main 
    - unzip it in `data/ftir`