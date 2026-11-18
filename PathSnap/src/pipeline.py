"""Shared helper: turns ONE trip folder into (track, nodes, arcs, labels, feature table)."""
import numpy as np
from src.core import load_trip, project_trip, label_points, generate_candidates
from src.features import build_features

def prepare_trip(part_folder, trip_id, radius=50.0, k_max=10, with_labels=True):
    track, nodes, arcs, route = load_trip(part_folder, trip_id)
    track, nodes = project_trip(track, nodes)
    cand = generate_candidates(track, nodes, arcs, radius=radius, k_max=k_max)
    feats = build_features(track, nodes, arcs, cand)
    feats["trip_id"] = trip_id
    labels = None
    if with_labels and route is not None:
        labels = label_points(track, nodes, arcs, route)
        feats["label"] = (feats.arc_id.values == labels[feats.point_idx.values]).astype(int)
    return track, nodes, arcs, labels, feats

def classifier_only_route(feats, proba):
    """Pick the highest-probability candidate for every GPS point."""
    f = feats.assign(p=proba)
    best = f.loc[f.groupby("point_idx").p.idxmax()]
    return dict(zip(best.point_idx.astype(int), best.arc_id.astype(int)))

def top1_accuracy(pred, labels):
    return float(np.mean([pred.get(i, -1) == labels[i] for i in range(len(labels))]))

def route_continuity(pred, arcs):
    seq = [pred[i] for i in sorted(pred)]
    seq = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
    if len(seq) < 2:
        return 1.0
    u, v = arcs.u.to_numpy(), arcs.v.to_numpy()
    return float(np.mean([v[a] == u[b] for a, b in zip(seq[:-1], seq[1:])]))
