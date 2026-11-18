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

### Create a Virtual Environment

```bash
python3.11 -m venv venv
```

### Activate the Environment

For macOS/Linux:

```bash
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

### Generate Features

```bash
python scripts/02_build_dataset.py
```

### Train and Compare Models

```bash
python scripts/03_train_models.py
```

### Perform Bayesian Optimisation

```bash
python scripts/04_bayes_opt.py
```

Run the scripts only after the required datasets and input feature files are available.

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

## 13. Project Status

Project implementation and evaluation are in progress.

Final model performance, optimisation results, and system evaluation will be updated after the official dataset is processed and experiments are completed.

## 14. Team Members

| Person | Name | Responsibility |
|---|---|---|
| Person A | Sakshi | Data Preparation |
| Person B | Ramitha | Machine Learning |
| Person C | Rushikesh | HMM and Route Reconstruction |
| Person D | Roopesh | API and User Interface |
