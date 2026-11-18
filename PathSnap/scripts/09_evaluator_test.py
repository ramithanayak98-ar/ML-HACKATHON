"""FINAL CHECK ONLY - run AFTER the model and beta were frozen.
Runs the final model on the evaluator-test trips, with and without the HMM, and saves the results.
Nothing in this script may be used to change the model, features or beta.
Run from the PathSnap folder:   python scripts/09_evaluator_test.py"""
import os, sys, time, joblib, pandas as pd
sys.path.insert(0, os.getcwd())
from src.pipeline import prepare_trip, classifier_only_route, top1_accuracy, route_continuity
from src.hmm import RoadGraph, viterbi
from src.features import FEATURES
from src.core import load_trip

EVAL_DIR = os.environ.get("EVAL_DIR", "data/eval_test")
BETA = 50.0
model = joblib.load("models/final_model.joblib")
trips = sorted(d for d in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR, d)))
print(f"{len(trips)} trips in {EVAL_DIR}: {trips}")

rows = []
for tid in trips:
    t0 = time.time()
    track, nodes, arcs, _, feats = prepare_trip(EVAL_DIR, tid, with_labels=False)   # prediction: no ground truth
    proba = model.predict_proba(feats[FEATURES])[:, 1]
    cls = classifier_only_route(feats, proba)
    hmm = viterbi(track, feats, proba, RoadGraph(nodes, arcs), beta=BETA)
    secs = time.time() - t0
    _, _, _, labels, _ = prepare_trip(EVAL_DIR, tid, with_labels=True)              # scoring: after prediction
    ref = set(load_trip(EVAL_DIR, tid)[3].tolist())
    row = dict(trip_id=tid, n_points=len(track), time_s=round(secs, 2))
    for name, pred in [("cls", cls), ("hmm", hmm)]:
        p = set(pred.values())
        row[f"{name}_top1"] = top1_accuracy(pred, labels)
        row[f"{name}_continuity"] = route_continuity(pred, arcs)
        row[f"{name}_arc_precision"] = len(ref & p) / max(len(p), 1)
        row[f"{name}_arc_recall"] = len(ref & p) / max(len(ref), 1)
    rows.append(row)
    print(f"{tid}: {row['n_points']:5d} pts | Top-1 {row['cls_top1']:.3f} -> {row['hmm_top1']:.3f} | "
          f"continuity {row['cls_continuity']:.3f} -> {row['hmm_continuity']:.3f} | {secs:.1f}s")

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.round(4).to_csv("results/evaluator_test_results.csv", index=False)
cols = [c for c in df.columns if c not in ("trip_id",)]
print("\nAVERAGE OVER TRIPS (classifier only -> + HMM, beta = 50)")
for m in ["top1", "continuity", "arc_precision", "arc_recall"]:
    print(f"  {m:14s} {df[f'cls_{m}'].mean()*100:6.2f}%  ->  {df[f'hmm_{m}'].mean()*100:6.2f}%")
pooled_cls = sum(r["cls_top1"] * r["n_points"] for r in rows) / df.n_points.sum()
pooled_hmm = sum(r["hmm_top1"] * r["n_points"] for r in rows) / df.n_points.sum()
print(f"  top1 (pooled over all {df.n_points.sum()} points)  {pooled_cls*100:.2f}%  ->  {pooled_hmm*100:.2f}%")
print("\nsaved results/evaluator_test_results.csv  (computed after the model was frozen)")
