# Brain Age Prediction with Longitudinal MRI

This repository contains the code for a thesis project focused on developing and evaluating machine learning models to predict changes in brain age using longitudinal brain imaging data.

The foundation for this work is based on the [LILAC model](https://github.com/heejong-kim/LILAC).

## Models

This project includes the following models:

1. **LILAC** – Reimplementation of the original [LILAC model](https://github.com/heejong-kim/LILAC).
2. **LILAC+** – An expanded version of LILAC with modifications and enhancements.
3. **Cross-sectional model** – A model of similar size and architecture to compare against the longitudinal models.

## Evaluation

All models are evaluated using a standardized evaluation pipeline to ensure consistency and reproducibility.

## Getting Started

In order ot activate the virtual environment, run the following code in the command line:
module load virtualenv/20.26.2-GCCcore-13.3.0 matplotlib/3.9.2-gfbf-2024a SciPy-bundle/2024.05-gfbf-2024a

Then activate the virtual environment by:
source my_venv/bin/activate