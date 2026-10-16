# Mazhai AI Crop

AI-assisted crop recommendation and crop-yield prediction project.

## Setup on Windows PowerShell

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

## Crop recommendation dataset

Download the Crop Recommendation dataset:
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Save its CSV as `data/crop_recommendation.csv`. Expected columns:
N, P, K, temperature, humidity, ph, rainfall, label.

Train:

    python train_crop_model.py

Run the web app:

    python -m streamlit run app.py

## Yield prediction

Add a separate historical dataset as `data/crop_yield.csv`.
It must contain a numeric `yield` column representing a documented yield unit.
Numeric columns will be used as predictors except ID, year, and target.
Check the dataset for data leakage and ensure the yield unit is consistent.

Train:

    python train_yield_model.py

The app currently provides the crop recommendation interface. Yield-model
training is included, but its inference interface still needs to be connected.

## Limitations

This is an educational prototype. Validate model performance and recommendations
with reliable local data and agricultural expertise before real-world use.
