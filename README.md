## TODO
- [ ] register datasets and download them from mlflow instead of manually
- [ ] rebuild images without datasets, document how this can be done on other devices
- [ ]
- [ ] figure out model deployment via standalone code. 
    - [ ] resolve warnings
    - [ ] add input/output schemas and descriptions
- [ ] improve evaluation, incl neater (in terms of mlflowui) per-class organisation, recall, precission, etc.
- [ ] improve training (LR scheduling and good optimizer)
    - [ ] pos weights implemented correctly?
- [ ] dataset registry and versioning
- [ ] implement and train fcg transformer
- [ ] visualize attention
- [ ] templates for reproducible dataset generation through augmentation of existing datasets in notebooks 
    - 2 kinds of transforms:
        - simple augmentations like adding noise, done at training time
        - theory-based augmentations to expand dataset at time of versioned dataset generation
- [ ] implement and train mt_fcg transformer
- [ ] clean up repo and improve docs


## Goals now
1. deployed model with defined multi-output dictionary
2. datasets via generating notebooks and mlflow
    - incl git hash tracking
    - single dataset class
3. proper training
4. logging not printing and artifact config upload to mlflow.
    - also log system stats to see if gpu optimized


## Datasets
### FTIR
- to use the FTIR dataset used in the FCG-Former paper:
    - download `dataset.zip` from https://huggingface.co/datasets/lycaoduong/FTIR/tree/main 
    - unzip it in `data/ftir`


## Docker
- `./Dockerfile.base` only installs dependencies
- `./Dockerfile.train` also copies over files. Used to build an image for a training job deployable to cloud.
- `.devcontainer/Dockerfile` also adds a non-root user and installs dev tools
- For the dev-container, MLFlow authentication environment variables are loaded from local `.env` or your shell
- For deploying jobs, will need to specify in the deployment command, along with hyparams

## Building docker images
- `make` build all 3 docker images. 
- can use args `build-base/build-train/build-dev` to only build some
- `make clean` removes the images