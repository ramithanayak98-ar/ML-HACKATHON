"""PERSON C - Steps C4-C6: classifier-only vs classifier+HMM on the student-test trips. Run:  python scripts/06_hmm_eval.py"""
import os, sys, json, time, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.getcwd())
from src.pipeline import prepare_trip, classifier_only_route, top1_accuracy, route_continuity
from src.hmm import RoadGraph, viterbi
from src.features import FEATURES
model = joblib.load(os.environ.get("MODEL", "models/final_model.joblib"))
BETAS = [float(b) for b in os.environ.get("BETAS", "2,5,10,20,50").split(",")]
rows = []
for tid in sorted(os.listdir("data/student_test")):
    track, nodes, arcs, labels, feats = prepare_trip("data/student_test", tid)
    p = model.predict_proba(feats[FEATURES])[:, 1]
    cls = classifier_only_route(feats, p); graph = RoadGraph(nodes, arcs)
    row = dict(trip_id=tid, n_points=len(track), cls_top1=top1_accuracy(cls, labels), cls_continuity=route_continuity(cls, arcs))
    for b in BETAS:
        t0 = time.time(); hm = viterbi(track, feats, p, graph, beta=b)
        row[f"hmm_top1_beta{b:g}"] = top1_accuracy(hm, labels); row[f"hmm_cont_beta{b:g}"] = route_continuity(hm, arcs)
        row[f"hmm_time_beta{b:g}"] = time.time() - t0
    rows.append(row); print(row)
df = pd.DataFrame(rows); df.to_csv("results/hmm_vs_classifier.csv", index=False)
print(df.drop(columns="trip_id").mean().round(4))
