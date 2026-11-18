"""PathSnap FastAPI backend (Laptop 1).
Run from the PathSnap folder:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Ground-truth routes are read ONLY here, and only AFTER a prediction is made, to score it."""
import os, sys, glob, json, time, logging, joblib, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline import prepare_trip, classifier_only_route, top1_accuracy, route_continuity
from src.hmm import RoadGraph, viterbi
from src.features import FEATURES
from src.core import load_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("pathsnap")
EVAL_DIR = os.environ.get("EVAL_DIR", "data/eval_test")      # the evaluator trips live ONLY on this laptop
BETA = float(os.environ.get("BETA", "50"))
FINAL = "final_model"

# load every saved model once at start-up (final_model = Optuna-tuned HistGB)
MODELS = {os.path.splitext(os.path.basename(p))[0]: joblib.load(p) for p in sorted(glob.glob("models/*.joblib"))}
log.info(f"loaded models: {list(MODELS)}  |  HMM beta = {BETA}")
app = FastAPI(title="PathSnap API")


class PredictRequest(BaseModel):
    trip_id: str
    model: str = FINAL        # which saved model to use
    use_hmm: bool = True      # False = classifier-only (best candidate per GPS point)


def _trip_ids():
    return sorted(d for d in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR, d)))


@app.get("/health")
def health():
    return {"status": "ok", "models": len(MODELS), "beta": BETA}


@app.get("/trips")
def trips():
    return {"trips": _trip_ids()}


@app.get("/models")
def models():
    return {"models": list(MODELS), "default": FINAL}


@app.get("/dev_results")
def dev_results():
    """Development-phase results (train CV + student-test trips) produced by scripts 03, 04 and 06.
    Contains NO evaluator-trip information."""
    out = {}
    if os.path.exists("results/model_comparison.csv"):
        out["models"] = pd.read_csv("results/model_comparison.csv").round(4).to_dict("records")
    if os.path.exists("results/tuned_vs_untuned.csv"):
        out["tuning"] = pd.read_csv("results/tuned_vs_untuned.csv").round(4).to_dict("records")
    if os.path.exists("results/optuna_best.json"):
        out["optuna_best"] = json.load(open("results/optuna_best.json"))
    hmm = {}
    for p in glob.glob("results/hmm_vs_classifier*.csv"):
        hmm.update(pd.read_csv(p).drop(columns="trip_id").mean().to_dict())
    if hmm:
        rows = [{"setting": "Classifier only", "top1": hmm.get("cls_top1"), "continuity": hmm.get("cls_continuity")}]
        betas = sorted({float(k.split("beta")[1]) for k in hmm if k.startswith("hmm_top1_beta")})
        rows += [{"setting": f"HMM beta={b:g}", "top1": hmm.get(f"hmm_top1_beta{b:g}"),
                  "continuity": hmm.get(f"hmm_cont_beta{b:g}")} for b in betas]
        out["hmm"] = rows
    return out


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in _trip_ids():
        raise HTTPException(404, "unknown trip")
    if req.model not in MODELS:
        raise HTTPException(404, f"unknown model; available: {list(MODELS)}")
    log.info(f"REQUEST trip={req.trip_id} model={req.model} hmm={req.use_hmm}")
    t0 = time.time()
    # 1) prediction - uses ONLY GPS + road network, never the reference route
    track, nodes, arcs, _, feats = prepare_trip(EVAL_DIR, req.trip_id, with_labels=False)
    proba = MODELS[req.model].predict_proba(feats[FEATURES])[:, 1]
    if req.use_hmm:
        pred = viterbi(track, feats, proba, RoadGraph(nodes, arcs), beta=BETA)
    else:
        pred = classifier_only_route(feats, proba)
    infer_s = time.time() - t0
    # 2) scoring - server side, AFTER prediction, with the hidden reference route
    metrics = dict(route_continuity=round(route_continuity(pred, arcs), 4),
                   inference_time_s=round(infer_s, 2), n_gps_points=len(track))
    route = load_trip(EVAL_DIR, req.trip_id)[3]
    if route is not None:
        _, _, _, labels, _ = prepare_trip(EVAL_DIR, req.trip_id, with_labels=True)
        ref_arcs, pred_arcs = set(route.tolist()), set(pred.values())
        metrics.update(point_top1_accuracy=round(top1_accuracy(pred, labels), 4),
                       route_arc_precision=round(len(ref_arcs & pred_arcs) / max(len(pred_arcs), 1), 4),
                       route_arc_recall=round(len(ref_arcs & pred_arcs) / max(len(ref_arcs), 1), 4))
    log.info(f"DONE trip={req.trip_id} model={req.model} hmm={req.use_hmm}: {metrics}")
    # 3) response: GPS + predicted route geometry + metrics (NO ground-truth geometry is sent)
    seq = [pred[i] for i in sorted(pred)]
    seq = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
    lines = [[[nodes.lat[arcs.u[a]], nodes.lon[arcs.u[a]]], [nodes.lat[arcs.v[a]], nodes.lon[arcs.v[a]]]] for a in seq]
    return {"trip_id": req.trip_id, "model": req.model, "use_hmm": req.use_hmm, "metrics": metrics,
            "gps": track[["lat", "lon"]].values[::2].tolist(), "predicted_route": lines}
