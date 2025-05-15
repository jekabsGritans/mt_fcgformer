## TODO

Finish this today/tmrw
-----------------------------------------------------
- [ ] finish model deployment
    - [ ] pass dependencies and transforms to log_model
    - [ ] test deployment and loading outside of this repo

- [ ] convert dataset to pandas

- [ ] improve training (LR scheduling and good optimizer)
    - [ ] implement pos weights computation

- [ ] train IRCnn locally

- [ ] improve evaluation, incl neater (in terms of mlflowui) per-class organisation, recall, precission and visualizations (confusion mat)

------------------------------------------------------

- [ ] implement FCGFormer and try to train locally

- [ ] extra logs (mass edit)
    - [ ] logs instead of prints
    - [ ] saving to file and uploading artifact
    - [ ] resource usage logs

- [ ] separate prediction heads and try to train locally

- [ ] visualize attention

- [ ] cloud deployment scripts + hyperparam search algos
- [ ] model improvement

------------------------------------------------------

- [ ] mass fix docstrings and write readme



## Datasets
- 2 kinds of transforms:
    - simple augmentations like adding noise, done at training time
    - theory-based augmentations to expand dataset at time of versioned dataset generation

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