## TODO
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


## Datasets
### FTIR
- to use the FTIR dataset used in the FCG-Former paper:
    - download `dataset.zip` from https://huggingface.co/datasets/lycaoduong/FTIR/tree/main 
    - unzip it in `data/ftir`