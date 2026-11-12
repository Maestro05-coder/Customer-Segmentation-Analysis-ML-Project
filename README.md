# Customer Segmentation Dashboard

Interactive Streamlit app built on the analysis in `customerSegmentation.py`:
RFM feature engineering, K-Means segmentation, PCA projection, segment
profiles, and revenue attribution.

## Run locally

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. `transactions.csv` must stay in the same
folder as `app.py`.

## Deploy for free on Streamlit Community Cloud (recommended)

Streamlit Community Cloud is free for public apps and needs no server setup.

1. **Put the app in a GitHub repository.** From this folder:

   ```bash
   git init
   git add app.py transactions.csv requirements.txt README.md .streamlit/config.toml .gitignore
   git commit -m "Streamlit customer segmentation app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

   Only these files are needed to deploy. `.venv/` and `plots/` are not
   required; `.gitignore` already excludes the virtualenv.

2. Go to <https://share.streamlit.io> and sign in with GitHub.

3. Click **Create app** → **Deploy a public app from GitHub**.

4. Fill in:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`

5. (Optional) Open **Advanced settings** and pick a Python version. Python
   3.12 or 3.13 is a safe choice; the app only needs the packages listed in
   `requirements.txt`.

6. Click **Deploy**. Streamlit installs `requirements.txt`, starts the app,
   and gives you a public URL like
   `https://<your-app-name>.streamlit.app`.

Notes:

- The GitHub repository must be **public** on the free tier.
- Free apps **sleep after a period of inactivity** and wake when someone
  visits the URL (the first load takes a little longer).
- Free resource limits are modest. This dataset is small (~44k rows), so the
  app fits comfortably; the pipeline results and K-Means fits are cached after
  the first run.
- To update the deployed app, just `git push` to the same branch.
- Never commit secrets or private customer data. `transactions.csv` here is
  synthetic, so it is safe to publish.

## Alternative: Hugging Face Spaces (also free)

1. Create a new Space at <https://huggingface.co/new-space> and choose the
   **Streamlit** SDK.
2. Upload `app.py`, `transactions.csv`, and `requirements.txt`.
3. Hugging Face detects Streamlit from the Space metadata and builds the app.
4. The free CPU tier is enough for this workload. Apps sleep after inactivity
   unless you upgrade.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard (entry point) |
| `transactions.csv` | Raw synthetic transaction data |
| `requirements.txt` | Python dependencies installed on deploy |
| `.streamlit/config.toml` | App theme |
| `customerSegmentation.py` | Original notebook-style analysis script |
| `syntethic_data.py` | Regenerates `transactions.csv` |
| `plots/`, `customer_segments.csv` | Outputs from the original script |
