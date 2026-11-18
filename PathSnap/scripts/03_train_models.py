"""PERSON B - Step B4: train and compare >=5 models. Run:  python scripts/03_train_models.py"""
import os, sys, time, json, numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from src.features import FEATURES
SEED = 42
train = pd.read_parquet("results/features_train.parquet")
test = pd.read_parquet("results/features_student_test.parquet")
MODELS = {   # name: (family, estimator, max training rows - slow models get a sample)
 "LogReg":  ("Linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), None),
 "GaussNB": ("Probabilistic", GaussianNB(), None),
 "KNN":     ("Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance")), 100_000),
 "DecTree": ("Tree", DecisionTreeClassifier(max_depth=12, random_state=SEED), None),
 "RandForest": ("Ensemble", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED), 600_000),
 "HistGB":  ("Boosting", HistGradientBoostingClassifier(random_state=SEED), None),
 "MLP":     ("Neural net", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=SEED)), 400_000),
}
def sample(df, n):
    return df if n is None or len(df) <= n else df.sample(n, random_state=SEED)
def top1(df, p):
    d = df[["point_idx", "trip_id", "label"]].assign(p=p)
    best = d.loc[d.groupby(["trip_id", "point_idx"]).p.idxmax()]
    return best.label.mean()          # fraction of points whose best candidate is the true arc
ONLY = os.environ.get("ONLY")          # e.g.  set ONLY=LogReg,HistGB  to run just some models
if ONLY: MODELS = {k: v for k, v in MODELS.items() if k in ONLY.split(",")}
rows = []; gkf = GroupKFold(n_splits=5)
for name, (family, est, nmax) in MODELS.items():
    cv_f1 = []
    for k, (tr_i, va_i) in enumerate(gkf.split(train, train.label, train.trip_id)):
        tr = sample(train.iloc[tr_i], nmax)
        va = sample(train.iloc[va_i], 50_000) if name == "KNN" else train.iloc[va_i]   # KNN prediction is slow
        est.fit(tr[FEATURES], tr.label); cv_f1.append(f1_score(va.label, est.predict(va[FEATURES]), average="macro"))
    t0 = time.time(); tr = sample(train, nmax); est.fit(tr[FEATURES], tr.label); train_time = time.time() - t0
    t0 = time.time(); p = est.predict_proba(test[FEATURES])[:, 1]; inf_time = time.time() - t0
    yhat = (p >= 0.5).astype(int)
    rows.append(dict(model=name, family=family, train_rows=len(tr), cv_macro_f1_mean=np.mean(cv_f1), cv_macro_f1_std=np.std(cv_f1),
        test_macro_f1=f1_score(test.label, yhat, average="macro"), test_precision=precision_score(test.label, yhat),
        test_recall=recall_score(test.label, yhat), test_log_loss=log_loss(test.label, np.clip(p, 1e-6, 1 - 1e-6)),
        test_top1=top1(test, p), train_time_s=train_time, inference_time_s=inf_time))
    print(rows[-1])
    joblib.dump(est, f"models/{name}.joblib")
    ConfusionMatrixDisplay.from_predictions(test.label, yhat); plt.title(name); plt.savefig(f"results/cm_{name}.png", dpi=120); plt.close()
    if name == "MLP":   # only the MLP has an epoch-wise history
        mlp = est[-1]; plt.plot(mlp.loss_curve_, label="train loss"); plt.plot(1 - np.array(mlp.validation_scores_), label="1 - val accuracy")
        plt.xlabel("epoch"); plt.legend(); plt.savefig("results/mlp_training_curve.png", dpi=120); plt.close()
res = pd.DataFrame(rows)
out = "results/model_comparison.csv"
if ONLY and os.path.exists(out):           # merge with results from earlier runs / other laptops
    old = pd.read_csv(out); res = pd.concat([old[~old.model.isin(res.model)], res], ignore_index=True)
res.to_csv(out, index=False); print(res.to_string())
for col in ["cv_macro_f1_mean", "test_macro_f1", "test_top1", "test_log_loss"]:
    res.plot.bar(x="model", y=col, legend=False, title=col); plt.tight_layout(); plt.savefig(f"results/compare_{col}.png", dpi=120); plt.close()
