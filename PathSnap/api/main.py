"""PERSON D - Laptop 1: FastAPI backend.
Run from the project folder:  uvicorn api.main:app --host 0.0.0.0 --port 8000"""
import os, sys, time, logging, joblib, numpy as np
sys.path.insert(0, os.getcwd())
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline import prepare_trip, top1_accuracy, route_continuity
from src.hmm import RoadGraph, viterbi
from src.features import FEATURES
from src.core import load_trip
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s"); log = logging.getLogger("pathsnap")
EVAL_DIR = os.environ.get("EVAL_DIR", "data/eval_test")      # the 10 evaluator trips live ONLY on this laptop
BETA = float(os.environ.get("BETA", "50"))
model = joblib.load("models/final_model.joblib")
app = FastAPI(title="PathSnap API")

class PredictRequest(BaseModel):
    trip_id: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/trips")
def trips():
    return {"trips": sorted(os.listdir(EVAL_DIR))}

@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in os.listdir(EVAL_DIR):
        raise HTTPException(404, "unknown trip")
    log.info(f"REQUEST received for trip {req.trip_id}")
    t0 = time.time()
    # 1) prediction pipeline - uses ONLY GPS + road network, never the reference route
    track, nodes, arcs, _, feats = prepare_trip(EVAL_DIR, req.trip_id, with_labels=False)
    proba = model.predict_proba(feats[FEATURES])[:, 1]
    pred = viterbi(track, feats, proba, RoadGraph(nodes, arcs), beta=BETA)
    infer_s = time.time() - t0
    # 2) evaluation - done server side, AFTER prediction, with the hidden reference route
    metrics = dict(route_continuity=round(route_continuity(pred, arcs), 4), inference_time_s=round(infer_s, 2), n_gps_points=len(track))
    route = load_trip(EVAL_DIR, req.trip_id)[3]
    if route is not None:                      # reference route present on the server -> score it
        _, _, _, labels, _ = prepare_trip(EVAL_DIR, req.trip_id, with_labels=True)
        ref_arcs, pred_arcs = set(route.tolist()), set(pred.values())
        metrics.update(point_top1_accuracy=round(top1_accuracy(pred, labels), 4),
                       route_arc_precision=round(len(ref_arcs & pred_arcs) / max(len(pred_arcs), 1), 4),
                       route_arc_recall=round(len(ref_arcs & pred_arcs) / max(len(ref_arcs), 1), 4))
    log.info(f"DONE trip {req.trip_id}: {metrics}")
    # 3) response: GPS + predicted route geometry + metrics (NO ground truth geometry is sent)
    seq = [pred[i] for i in sorted(pred)]; seq = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
    lines = [[[nodes.lat[arcs.u[a]], nodes.lon[arcs.u[a]]], [nodes.lat[arcs.v[a]], nodes.lon[arcs.v[a]]]] for a in seq]
    return {"trip_id": req.trip_id, "metrics": metrics, "gps": track[["lat", "lon"]].values[::2].tolist(), "predicted_route": lines}
