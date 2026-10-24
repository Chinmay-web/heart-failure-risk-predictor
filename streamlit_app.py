"""Educational exploration of historical heart-failure follow-up outcomes."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Failure Risk Explorer", page_icon="🫀", layout="centered")

@st.cache_resource
def load_bundle():
    return joblib.load(Path(__file__).with_name("model.joblib"))

bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]
medians = bundle["medians"]
threshold = float(bundle.get("threshold", 0.5))

st.title("🫀 Heart Failure Risk Explorer")
st.caption("Machine Learning Dashboard | Educational research demonstration")
st.warning(
    "Not medical advice. This model uses a small historical observational dataset "
    "(299 patients with heart failure). It estimates a recorded death event during "
    "dataset follow-up, not an individual's medical risk or prognosis. "
    "Never use these outputs for clinical decisions."
)
st.write("Explore how hypothetical inputs affect a machine-learning model's output. "
         "Other clinical features are filled using training-set medians.")

profiles = {
    "Example 1 — Typical inputs": (60, 38, 1.1, 137, False, False, False, False),
    "Example 2 — Altered inputs": (75, 25, 2.1, 130, True, True, True, False),
    "Example 3 — Different inputs": (45, 55, 0.9, 140, False, False, False, False),
}
choice = st.selectbox("Example profile", list(profiles))
age0, ef0, cr0, na0, bp0, an0, db0, sm0 = profiles[choice]

with st.form("prediction"):
    age = st.slider("Age (years)", 40, 95, int(age0))
    ef = st.slider("Ejection fraction (%)", 10, 80, int(ef0))
    cr = st.slider("Serum creatinine (mg/dL)", 0.5, 9.5, float(cr0), 0.1)
    na = st.slider("Serum sodium (mEq/L)", 113, 148, int(na0))
    bp = st.checkbox("High blood pressure", value=bp0)
    an = st.checkbox("Anaemia", value=an0)
    db = st.checkbox("Diabetes", value=db0)
    sm = st.checkbox("Smoker", value=sm0)
    submit = st.form_submit_button("Explore model output")

if submit:
    row = dict(medians)
    row.update(age=age, ejection_fraction=ef, serum_creatinine=cr,
               serum_sodium=na, high_blood_pressure=int(bp),
               anaemia=int(an), diabetes=int(db), smoking=int(sm))
    probability = float(model.predict_proba(pd.DataFrame([row])[features])[0, 1])
    st.metric("Estimated probability of recorded event", f"{probability:.1%}")
    st.progress(max(0.0, min(1.0, probability)))
    st.write(f"Demonstration threshold: **{threshold:.0%}** — "
             f"**{'above' if probability >= threshold else 'below'} threshold**.")
    st.info("This is an educational model output, not a clinical alert or "
            "individual prognosis. Changing inputs demonstrates model behavior, "
            "not causation.")

st.divider()
st.caption("Dataset: UCI Heart Failure Clinical Records. "
           "Chicco & Jurman (2020), BMC Medical Informatics and Decision Making. "
           "Model is not externally clinically validated.")
