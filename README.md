# Brain Age Prediction with Longitudinal MRI

This repository contains the code for a thesis project focused on developing and evaluating machine learning models to predict changes in brain age using longitudinal brain imaging data.

The foundation for this work is based on the [LILAC model](https://github.com/heejong-kim/LILAC).

## Models

This project includes the following models:

1. **LILAC** – Reimplementation of the original [LILAC model](https://github.com/heejong-kim/LILAC).
2. **LILAC+** – An expanded version of LILAC with modifications and enhancements.
3. **Cross-sectional model** – A model of similar size and architecture to compare against the longitudinal models.
4. **Autoencoder model** - The MRIs are first passed through the autoencoder [MedVAE](https://github.com/StanfordMIMI/MedVAE) before being passed through LILAC.

## Evaluation

All models are evaluated using a standardized evaluation pipeline to ensure consistency and reproducibility with the evaluation_pipeline.py script found in tools.

## Getting Started

In order ot activate the virtual environment, run the following code in the command line:
module load virtualenv/20.26.2-GCCcore-13.3.0 matplotlib/3.9.2-gfbf-2024a SciPy-bundle/2024.05-gfbf-2024a

Then activate the virtual environment by:
source my_venv/bin/activate

## Training/Applying/Evaluating the Models

The models are trained with run_CV_LILAC.py (LILAC and LILAC+), run_CS_CNN.py (cross-sectional model) and run_AE.py (autoencoder model) found in scripts. The respective jobscripts can be found in jobscripts.

After the models have been trained, they can be applied to arbitrary participants using the respective apply_*.py scripts found under scripts. Here, the predicted values as well as any relevant metadata is output together as a .csv file.

All validation results can be concatenated together using the CV_evaluation.py script found under scripts in order to get a full performance overview of the models on unseen data.

After the models have been applied, their performance can be evaluated by applying the evaluation_pipeline.py script found under tools.

## General Workflow

1. Train desired model.
2. Apply model to all validation folds.
3. Cocatenate validation results using CV_evaluation.py
4. Evaluate performance using evaluation_pipeline.py

