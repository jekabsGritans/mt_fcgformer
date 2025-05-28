## TODO


- [ ] implement FCGFormer together with pandas conversion
- [ ] convert dataset to pandas
    - supported column types: spectra, categorical, boolean, numerical
    - transform only applied to spectrum pre-tokenization
    - other are batched as python primitives (or myb ndarray)
    - cnn ignores rest
    - transformer does mixed-input tokenization within model
        - should keep whatever is fixed in gpu memory (is embedding learned?)
    - need to provide inputs and targets as lists of column names (verify unique)

- [ ] explicit train test setting within notebook pre-augmentation? or maybe validation is ok on augmented

- [ ] train transformer

- [ ] attention visualisation regions as additional output
------------------------------------------------------

- [ ] separate prediction heads and try to train locally

- [ ] visualize attention
    - [ ] separate visualisation for spectral regions and flags

-------------------------------------------------------

- [ ] cloud deployment scripts + hyperparam search algos

------------------------------------------------------

- [ ] fix some tokens (if doable)
- [ ] more analytics (e.g. see increase in accuracy based on fixing some targets)

------------------------------------------------------
- [ ] mass fix docstrings and write readme


## Spectrum handling
- for training / dataset, we have already-interpolated spectra of wavenum 400-4000 cm^-1 length 3602 (as in FCGFormer paper)

## Attention visualisation
- the transformer resizes spectrum to 1024 and then splits it into 64 patches of 16

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
