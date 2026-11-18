"""PERSON B - Step B6: Bayesian hyperparameter optimisation (Optuna TPE). Run:  python scripts/04_bayes_opt.py"""
import os, sys, time, json, numpy as np, pandas as pd, joblib, optuna
sys.path.insert(0, os.getcwd())
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import FEATURES
SEED = 42; N_TRIALS = int(os.environ.get("N_TRIALS", "25"))
SEARCH_ROWS = int(os.environ.get("SEARCH_ROWS", "300000"))   # rows used inside the search (speed); final refit uses ALL rows
N_FOLDS = int(os.environ.get("N_FOLDS", "3"))
train = pd.read_parquet("results/features_train.parquet"); test = pd.read_parquet("results/features_student_test.parquet")
gkf = GroupKFold(n_splits=N_FOLDS)
search = train.sample(min(SEARCH_ROWS, len(train)), random_state=SEED).reset_index(drop=True)
def objective(trial):
    params = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                  max_iter=trial.suggest_int("max_iter", 100, 600),
                  max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
                  min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 500, log=True),
                  l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10, log=True))
    scores = []
    for tr_i, va_i in gkf.split(search, search.label, search.trip_id):
        m = HistGradientBoostingClassifier(random_state=SEED, **params).fit(search.iloc[tr_i][FEATURES], search.iloc[tr_i].label)
        scores.append(f1_score(search.iloc[va_i].label, m.predict(search.iloc[va_i][FEATURES]), average="macro"))
    return float(np.mean(scores))
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time(); study.optimize(objective, n_trials=N_TRIALS); cost = time.time() - t0
study.trials_dataframe().to_csv("results/optuna_trials.csv", index=False)
json.dump(dict(best_params=study.best_params, best_cv_macro_f1=study.best_value, n_trials=N_TRIALS, search_rows=len(search), n_folds=N_FOLDS, search_time_s=cost),
          open("results/optuna_best.json", "w"), indent=2)
best = HistGradientBoostingClassifier(random_state=SEED, **study.best_params).fit(train[FEATURES], train.label)
joblib.dump(best, "models/final_model.joblib"); json.dump(FEATURES, open("models/features.json", "w"))
print("best", study.best_params, study.best_value, f"{cost:.0f}s")
# ---- tuned vs untuned: SAME features, SAME CV folds, SAME student-test trips ----
def evaluate(name, params):
    cv = []
    for tr_i, va_i in gkf.split(search, search.label, search.trip_id):   # same rows + folds as the search
        m = HistGradientBoostingClassifier(random_state=SEED, **params).fit(search.iloc[tr_i][FEATURES], search.iloc[tr_i].label)
        cv.append(f1_score(search.iloc[va_i].label, m.predict(search.iloc[va_i][FEATURES]), average="macro"))
    m = HistGradientBoostingClassifier(random_state=SEED, **params).fit(train[FEATURES], train.label)
    t0 = time.time(); p = m.predict_proba(test[FEATURES])[:, 1]; inf = time.time() - t0
    d = test[["trip_id", "point_idx", "label"]].assign(p=p); top1 = d.loc[d.groupby(["trip_id", "point_idx"]).p.idxmax()].label.mean()
    return dict(model=name, cv_macro_f1=np.mean(cv), test_top1=top1, test_macro_f1=f1_score(test.label, p >= 0.5, average="macro"),
                test_log_loss=log_loss(test.label, np.clip(p, 1e-6, 1 - 1e-6)), inference_time_s=inf)
cmp = pd.DataFrame([evaluate("HistGB untuned (defaults)", {}), evaluate("HistGB tuned (Optuna)", study.best_params)])
cmp.to_csv("results/tuned_vs_untuned.csv", index=False); print(cmp.to_string())
