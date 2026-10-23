"""Interactive educational explorer for the heart-failure follow-up dataset."""
import joblib
import pandas as pd
import gradio as gr

bundle = joblib.load("model.joblib")
model, features, medians = bundle["model"], bundle["features"], bundle["medians"]
DEMO_THRESHOLD = bundle.get("threshold", 0.5)

DISCLAIMER = (
    "**Educational research demo only.** Trained on a small observational dataset "
    "of 299 patients. The output is an estimated probability of the recorded "
    "death event during the dataset's follow-up period—not a diagnosis, "
    "personal medical risk assessment, or treatment recommendation. "
    "Do not use for clinical decisions."
)

def predict_risk(age, ejection_fraction, serum_creatinine, serum_sodium,
                 high_blood_pressure, anaemia, diabetes, smoking):
    """Predict a hypothetical example using training-set medians for hidden inputs."""
    row = dict(medians)
    row.update(
        age=age,
        ejection_fraction=ejection_fraction,
        serum_creatinine=serum_creatinine,
        serum_sodium=serum_sodium,
        high_blood_pressure=int(high_blood_pressure),
        anaemia=int(anaemia),
        diabetes=int(diabetes),
        smoking=int(smoking),
    )
    probability = float(model.predict_proba(pd.DataFrame([row])[features])[0, 1])
    flagged = probability >= DEMO_THRESHOLD
    demo_label = "Above" if flagged else "Below"
    summary = (
        f"**Demonstration threshold:** {DEMO_THRESHOLD:.0%} — {demo_label.lower()} threshold. "
        "This is a teaching illustration, not a clinical alert or recommendation.\n\n"
        f"### Estimated dataset outcome probability: {probability:.1%}\n"
        "This estimate applies only to the historical dataset's recorded outcome "
        "and follow-up period. It is **not** an individual prognosis.\n\n"
        "**Modeling note:** Other model features are filled with training-set "
        "medians. Changing inputs demonstrates model behavior, not causation.\n\n"
        + DISCLAIMER
    )
    return {"Recorded death event": probability, "No recorded death event": 1 - probability}, summary

EXAMPLES = [
    [60, 38, 1.1, 137, False, False, False, False],
    [75, 25, 2.1, 130, True, True, True, False],
    [45, 55, 0.9, 140, False, False, False, False],
]

demo = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Slider(40, 95, value=60, step=1, label="Age (years)"),
        gr.Slider(10, 80, value=38, step=1, label="Ejection fraction (%)"),
        gr.Slider(0.5, 9.5, value=1.1, step=0.1, label="Serum creatinine (mg/dL)"),
        gr.Slider(113, 148, value=137, step=1, label="Serum sodium (mEq/L)"),
        gr.Checkbox(label="High blood pressure"),
        gr.Checkbox(label="Anaemia"),
        gr.Checkbox(label="Diabetes"),
        gr.Checkbox(label="Smoker"),
    ],
    outputs=[gr.Label(num_top_classes=2, label="Estimated outcome probabilities"), gr.Markdown()],
    examples=EXAMPLES,
    title="Heart Failure Risk Explorer | Educational AI Demo",
    description=(
        "Explore how a machine-learning model responds to **hypothetical** patient "
        "features. Try an example profile or adjust the controls. "
        "The target is the dataset's recorded death event during follow-up. "
        "The displayed cutoff is for educational comparison only.\n\n"
        + DISCLAIMER
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
