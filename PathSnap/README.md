# PathSnap – Map-Matching & Route Reconstruction (UE24CS352A Hackathon)

Noisy GPS points → candidate road segments → ML classifier → HMM/Viterbi → FastAPI (Laptop 1) → Streamlit UI (Laptop 2).

## Folder layout
```
PathSnap/
  data/            (NOT in git) train/, student_test/, eval_test/  – one sub-folder per trip
  src/             core.py (loading, projection, labels, candidates), features.py, hmm.py, pipeline.py
  scripts/         01_eda.py  02_build_dataset.py  03_train_models.py  04_bayes_opt.py  06_hmm_eval.py  07_make_maps.py
  api/main.py      FastAPI backend (Laptop 1)
  ui/app.py        Streamlit evaluator UI (Laptop 2)
  models/          final_model.joblib (+ other models, not committed)
  results/         CSV tables, plots, Optuna trials
  maps/            Folium HTML maps
```

## Install (Windows, Command Prompt)
```
cd PathSnap
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the full pipeline (from the PathSnap folder, venv active)
```
python scripts/01_eda.py                 # EDA tables, plots, Folium map
python scripts/02_build_dataset.py       # candidate + feature tables (about 1 minute)
python scripts/03_train_models.py        # 7 models, GroupKFold CV + student-test comparison
python scripts/04_bayes_opt.py           # Optuna (TPE) search -> models/final_model.joblib
python scripts/06_hmm_eval.py            # classifier-only vs classifier+HMM
python scripts/07_make_maps.py 00000000  # Folium map for one student-test trip
```

## Deployment (two laptops on the same network)
Laptop 1 (API): put the 10 evaluation trips in `data/eval_test/`, then
```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Laptop 2 (UI):
```
set API_URL=http://<LAPTOP-1-IPv4>:8000
streamlit run ui/app.py
```
Ground-truth routes stay on Laptop 1; the API returns only the predicted route and metrics.

## AI-assistance disclosure
Parts of the starter code were generated with an AI assistant (Claude) and then run, checked and modified by the team. All reported numbers come from our own runs of this repository.
