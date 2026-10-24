# Deploy on Streamlit Community Cloud

1. Upload `streamlit_app.py`, `model.joblib`, `requirements.txt`, and `.python-version` to the root of your existing public GitHub repository.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app** / **Deploy a public app from GitHub**.
4. Select your repository, branch `main`, and main file `streamlit_app.py`.
5. Deploy and wait for installation. Use Python 3.11 if the hosting UI offers a Python version selection.
6. Test each example profile and verify the medical disclaimer appears.

The trained model was saved with scikit-learn 1.6.1, so the dependency is pinned to that version. Never load untrusted joblib files.
