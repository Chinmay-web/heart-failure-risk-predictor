# Heart Failure Risk Explorer 🫀

**Educational machine-learning demo** by Chinmay Manakkattil. **Not for clinical use.**

An interactive Gradio application exploring how classification models respond to hypothetical clinical measurements from the UCI Heart Failure Clinical Records dataset.

## Features

- Dataset quality checks, feature summaries, and visualizations
- Logistic Regression, Random Forest, and Gradient Boosting comparison
- Five-fold stratified cross-validation on the training split
- Held-out ROC-AUC, confusion matrix, precision, recall, and F1 evaluation
- Threshold trade-off analysis using out-of-fold training predictions
- Interactive Gradio interface with example profiles and clear safety warnings

## Data and interpretation

Uses the UCI Heart Failure Clinical Records dataset (299 records). The label represents a recorded death event **during follow-up**; it does not establish a general, future individual risk. The `time` follow-up variable is excluded from the final predictor because it would not be available at prediction time. Other non-displayed features are filled with training-set medians in the demo.

Dataset: [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records). See Chicco and Jurman (2020) for discussion of the dataset.

## Reproduce training

1. Open `01_Intermediate_heart_failure_risk_app.ipynb` in Google Colab.
2. Choose **Runtime → Run all**. The notebook fetches the dataset using `ucimlrepo` (or accepts a local CSV if unavailable).
3. Inspect the training-only cross-validation comparison, test metrics, and threshold trade-offs.
4. Download the generated `model.joblib` from Colab. The app cannot start without this artifact.

## Run the app locally

Place `model.joblib` alongside `app.py` and `requirements.txt`, then run:

```bash
python -m pip install -r requirements.txt
python app.py
```

For a hosted demo, create a Gradio Space on Hugging Face and upload `app.py`, `requirements.txt`, and the generated `model.joblib`. Test the hosted app before sharing a link.

## Evaluation results

**Results pending a verified full notebook run.** Insert the actual test ROC-AUC, training CV ROC-AUC, recall, and precision from Colab here; do not report unverified numbers.

## Limitations and safety

- Small observational sample; no external or prospective validation
- Predicted scores may be uncalibrated and may not generalize to other populations
- Chosen 0.40 demonstration threshold is **not** a clinical cutoff
- Associations are not causal effects
- Do **not** use for diagnosis, triage, treatment, or decisions about any patient

## Tech stack

Python · pandas · scikit-learn · Matplotlib · Gradio · Google Colab
