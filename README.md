# PathSnap – GPS Map Matching Using Machine Learning

## 1. Project Overview

PathSnap is a GPS map-matching system that converts noisy GPS location data into realistic routes on a road network.

GPS devices sometimes produce inaccurate location points due to signal interference, buildings, tunnels, and other environmental conditions.

PathSnap uses machine learning and Hidden Markov Models (HMM) to identify the most likely road segments for a sequence of GPS points and reconstruct a continuous route.

## 2. Problem Statement

Raw GPS data may contain inaccurate coordinates, making it difficult to identify the actual roads travelled by a vehicle.

The objective of PathSnap is to:
- Process noisy GPS trajectories.
- Identify possible road segments for each GPS point.
- Use machine learning to score candidate road segments.
- Apply HMM and Viterbi decoding to find the most likely continuous route.
- Display the reconstructed route through a user interface.

## 3. Objectives

- Improve GPS trajectory accuracy.
- Compare multiple machine learning classification models.
- Identify the best-performing model using evaluation metrics.
- Optimise model hyperparameters using Bayesian optimisation.
- Reconstruct realistic routes using HMM.
- Provide an interface to visualise map-matched routes.

## 4. System Workflow

1. **GPS Data Input:** Collect GPS trajectories.
2. **Data Preprocessing:** Clean and prepare GPS data.
3. **Feature Engineering:** Generate useful features from GPS points and candidate road segments.
4. **Candidate Generation:** Identify possible road segments for each GPS observation.
5. **Machine Learning:** Train and compare classification models.
6. **Bayesian Optimisation:** Tune the selected model's hyperparameters.
7. **HMM Map Matching:** Use sequence information and Viterbi decoding to reconstruct a continuous route.
8. **Visualisation:** Display the final route through the application.

## 5. Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Hyperparameter Optimisation:** Optuna
- **Visualisation:** Matplotlib, Folium
- **Backend API:** FastAPI
- **User Interface:** Streamlit
- **Model Storage:** Joblib
- **Version Control:** Git and GitHub

## 6. Machine Learning Models

The project compares the following seven classification models:

1. Logistic Regression
2. Gaussian Naive Bayes
3. K-Nearest Neighbours (KNN)
4. Decision Tree
5. Random Forest
6. Histogram-based Gradient Boosting
7. Multi-Layer Perceptron (MLP)

The models are evaluated to identify the most suitable model for GPS road-segment classification.

## 7. Model Evaluation Metrics

- **Precision:** Measures how many predicted positive results are correct.
- **Recall:** Measures how many actual positive results are identified.
- **Macro F1-score:** Measures the balance between precision and recall across classes.
- **Log Loss:** Measures the quality of predicted probabilities. Lower values are better.
- **Top-1 Accuracy:** Measures whether the highest-scoring candidate road segment is correct.
- **Route Continuity:** Evaluates the consistency of the reconstructed route.

GroupKFold cross-validation is used to prevent GPS observations from the same trip appearing in both training and validation groups.

## 8. Bayesian Optimisation

Optuna is used to optimise the hyperparameters of the selected machine learning model.

The optimisation process aims to improve the macro F1-score while maintaining good generalisation performance.

## 9. Project Structure

```text
PathSnap/
│
├── api/                 # FastAPI backend
├── ui/                  # Streamlit user interface
├── maps/                # Map visualisation outputs
├── models/              # Trained machine learning models
├── results/             # Model evaluation results
├── report/              # Project reports
├── scripts/             # Data processing and training scripts
├── src/                 # Core source code
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 10. Setup Instructions

### Prerequisites

- Python 3.11
- pip
- Git

### Clone the Repository

```bash
git clone https://github.com/ramithanayak98-ar/ML-HACKATHON.git
cd ML-HACKATHON/PathSnap
```

### Create and Activate a Virtual Environment

Windows (Command Prompt):

```bat
py -3.11 -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dataset Setup

Place the official datasets in the following structure:

```text
data/
├── train/
├── student_test/
└── eval_test/
```

Each directory should contain the trip folders/files in the format provided by the organisers.

The evaluation dataset must remain separate and must not be used for model training or tuning.

### Run the Full Pipeline (in order, from the `PathSnap` folder)

| Step | Command | Output |
|---|---|---|
| 1. Dataset exploration | `python scripts/01_eda.py` | `results/eda/` inventory + histograms, `maps/eda_trip_*.html` |
| 2. Candidates + features | `python scripts/02_build_dataset.py` | `results/features_*.parquet` (about 1 min) |
| 3. Train and compare 7 models | `python scripts/03_train_models.py` | `results/model_comparison.csv`, comparison plots, confusion matrices |
| 4. Bayesian optimisation | `python scripts/04_bayes_opt.py` | `results/optuna_*`, `results/tuned_vs_untuned.csv`, `models/final_model.joblib` |
| 5. Classifier vs HMM | `python scripts/06_hmm_eval.py` | `results/hmm_vs_classifier.csv` |
| 6. Route maps | `python scripts/07_make_maps.py 00000052` | `maps/route_00000052.html` |
| 7. Technical report | `pip install reportlab` then `python scripts/08_make_report.py` | `report/PathSnap_Technical_Report.pdf` |

Optional settings (Windows `set NAME=value`, macOS/Linux `export NAME=value`):
`ONLY=HistGB,LogReg` (train only some models), `N_TRIALS=25` (Optuna trials), `MODEL=models/HistGB.joblib` and `BETAS=2,5,10,20,50` (HMM evaluation).

## Deployment (Two-Laptop Evaluation)

**Laptop 1 - API (holds the model, road networks and the hidden reference routes)**

1. Put the 10 evaluation trips in `PathSnap/data/eval_test/` (Laptop 1 only).
2. Start the API from the `PathSnap` folder:
   ```bat
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   The window prints `loaded models: [...]  |  HMM beta = 50.0` and logs every request (`REQUEST ...`, `DONE ...`).
3. Find Laptop 1's address with `ipconfig` (IPv4 Address). If Windows Firewall blocks port 8000, run in an administrator Command Prompt:
   ```bat
   netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
   ```

**Laptop 2 - SnapRoute UI (needs only the code, no data and no model)**

```bat
set API_URL=http://<Laptop-1-IPv4>:8000
streamlit run ui/app.py
```

Open `http://localhost:8501`. The sidebar shows *API connected*. Choose a trip and press **Run map-matching**.
Both laptops must be on the same network (a phone hotspot works best).

**API endpoints:** `GET /health`, `GET /trips`, `GET /models`, `GET /dev_results`, `POST /predict {"trip_id": "...", "model": "final_model", "use_hmm": true}`.
Ground-truth routes are read only on Laptop 1, after the prediction, to score it; they are never sent to the UI.

## 11. Team Responsibilities

- **Person A – Sakshi:** GPS data preparation, data exploration, preprocessing, and feature generation.
- **Person B – Ramitha:** Machine learning model training, model comparison, evaluation, and Bayesian optimisation.
- **Person C – Rushikesh:** Hidden Markov Model implementation and route reconstruction.
- **Person D – Roopesh:** API development, user interface, and system integration.

## 12. Expected Deliverables

- Processed GPS feature datasets.
- Comparison of seven machine learning models.
- Model evaluation metrics and visualisations.
- Bayesian optimisation results.
- Final trained machine learning model.
- HMM-based reconstructed routes.
- API and user interface.
- Final project report and documentation.

## 13. Project Status and Results

All components are implemented and tested end to end. Results on the 10 student-test trips (full details in `report/PathSnap_Technical_Report.pdf`):

| Item | Result |
|---|---|
| Training pairs (GPS point x candidate road) | 1,228,708 (15% positive) |
| Candidate miss rate | 0.27% |
| Models compared | 7 (LogReg, GaussNB, KNN, DecTree, RandForest, HistGB, MLP) |
| Best student-test Top-1 (classifier only) | Random Forest 85.6%, HistGB 85.4% |
| Selected model | HistGradientBoosting, tuned with Optuna TPE (25 trials) |
| Tuned vs untuned HistGB | Top-1 85.44% -> 85.62%, CV macro F1 0.8967 -> 0.8985 |
| HMM decoder | beta = 50 (tuned on student-test trips) |
| Classifier only -> + HMM (trip average) | Top-1 87.3% -> 88.6%, route continuity 75.0% -> 87.4% |
| Evaluator UI | "SnapRoute" (Streamlit) + FastAPI backend |

## 14. Team Members

| Person | Name | Responsibility |
|---|---|---|
| Person A | Sakshi | Data Preparation |
| Person B | Ramitha | Machine Learning |
| Person C | Rushikesh | HMM and Route Reconstruction |
| Person D | Roopesh | API and User Interface |

## 15. AI-Assistance Disclosure

The source code (data processing, feature engineering, model training, Bayesian optimisation, HMM/Viterbi decoder, API and UI) and a first draft of the technical report were produced with an AI assistant (Claude, Anthropic). The team ran every experiment, chose the decoder setting from the measured results, and all reported numbers come from those runs.
