## TODO

Finish this today/tmrw
-----------------------------------------------------

- [ ] convert dataset to pandas
    - supported column types: spectra, categorical, boolean, numerical
    - transform only applied to spectrum pre-tokenization
    - other are batched as python primitives (or myb ndarray)
    - cnn ignores rest
    - transformer does mixed-input tokenization within model
        - should keep whatever is fixed in gpu memory (is embedding learned?)
    - need to provide inputs and targets as lists of column names (verify unique)

- [ ] improve training (LR scheduling and good optimizer)

- [ ] train IRCnn locally

------------------------------------------------------

- [ ] implement FCGFormer and try to train locally

- [ ] separate prediction heads and try to train locally

- [ ] visualize attention
    - [ ] separate visualisation for spectral regions and flags

- [ ] fix some tokens (if doable)

-------------------------------------------------------

- [ ] extra logs (mass edit)
    - [ ] logs instead of prints
    - [ ] saving to file and uploading artifact
    - [ ] resource usage logs

- [ ] cloud deployment scripts + hyperparam search algos
- [ ] model improvement

------------------------------------------------------

- [ ] mass fix docstrings and write readme

- [ ] performance eval
    - optimal threshold can be found by maximising f1 in CNN paper


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
