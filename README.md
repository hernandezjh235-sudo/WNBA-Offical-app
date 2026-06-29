# ONE WAY PICKZ — WNBA Prop Engine

Streamlit WNBA player prop projection app for PTS, REB, AST, and PRA.

## Included
- `app.py` — main Streamlit app
- `requirements.txt` — Python dependencies for Streamlit Cloud/GitHub deploy
- `.streamlit/config.toml` — dark theme + server config

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Upload this folder to GitHub.
2. In Streamlit Cloud, select the repo.
3. Main file path: `app.py`.
4. Deploy.

## Data/log files
The app creates its own `wnba_engine/` folder at runtime for caches, logs, manual lines, official picks, learning logs, and CSV exports.
