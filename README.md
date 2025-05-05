## TODO
- [ ] finish basic train/eval/predict with proper configs and scripts
- [ ] clean up configs (e.g. share device as top level config, etc.)
- [ ] integrate mlflow and more metrics / monitoring
- [ ] dockerize (first custom deployable, then myb dev container)
- [ ] implement fcg transformer

## Datasets
### FTIR
- to use the FTIR dataset used in the FCG-Former paper:
    - download `dataset.zip` from https://huggingface.co/datasets/lycaoduong/FTIR/tree/main 
    - unzip it in `data/ftir`