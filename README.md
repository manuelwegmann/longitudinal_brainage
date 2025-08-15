# Brain Age Prediction with Longitudinal MRI

This repository contains the code for a thesis project focused on developing and evaluating machine learning models to predict changes in brain age using longitudinal brain imaging data. Oasis-3 (https://sites.wustl.edu/oasisbrains/) was used as the brain MRI dataset for training and validation.

The foundation for this work is based on the [LILAC model](https://github.com/heejong-kim/LILAC).

## Models

This project includes the following models:

1. **LILAC** – Reimplementation of the original [LILAC model](https://github.com/heejong-kim/LILAC).
2. **LILAC+** – An expanded version of LILAC with modifications and enhancements.
3. **Cross-sectional model** – A model of similar size and architecture as LILAC to compare against the longitudinal models.
4. **Autoencoder model** - Similar architceture to LILAC, however features are not extracted via a Siamese Neural Network but with the autoencoder [MedVAE](https://github.com/StanfordMIMI/MedVAE).

## Getting Started

In order to activate the virtual environment, run the following code in the command line:
module load virtualenv/20.26.2-GCCcore-13.3.0 matplotlib/3.9.2-gfbf-2024a SciPy-bundle/2024.05-gfbf-2024a

Then activate the virtual environment by:
source my_venv/bin/activate

## Data Preperation and Cleaning
Some participants in the original dataset had missing age information for some scan sessions. These were reconstructed, where possible, with the generate_new_session_files.py script found under tools.

Custom dataloaders for the respective models are found under loader.py (LILAC/LILAC+), loader_CS,py (cross-sectional CNN) and loader_AE.py (autoencoder model). Latent representations via MedVAE for the autoencoder model were computed beforehand and saved under data.

## Workflow

All models are trained and evaluated via the same pipeline. Below is an example of full training and evaluation of LILAC.

1. Train LILAC via run_LILAC.py (scripts)
2. After completing training, make predictions on each validation fold via apply_LILAC.py (scripts)
3. Concatenate predictions on all folds for evaluation via concatenate_folds.py (evaluation)
4. Generate summary statistics of model performance via summary_statistics.py (evaluation)

For the cross-sectional model, longitudinal predictions have to be made after making prediction on the validation folds, this is done by CS_longitudinal.py (scripts).

For the autoencoder model, step 2 is redundant.


## Detailed evaluation

Further scripts for evaluating difference in model performance via statitical tests with regards to optional meta data that is included in the model can be found under evaluation.

