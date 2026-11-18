"""PERSON C - Step C7: Folium map with GPS, road network, reference route, predicted route. Run:  python scripts/07_make_maps.py 00000000"""
import os, sys, joblib, folium
sys.path.insert(0, os.getcwd())
from src.pipeline import prepare_trip
from src.hmm import RoadGraph, viterbi
from src.features import FEATURES
from src.core import load_trip
tid = sys.argv[1] if len(sys.argv) > 1 else sorted(os.listdir("data/student_test"))[0]
BETA = float(os.environ.get("BETA", "10"))
model = joblib.load("models/final_model.joblib")
track, nodes, arcs, labels, feats = prepare_trip("data/student_test", tid)
pred = viterbi(track, feats, model.predict_proba(feats[FEATURES])[:, 1], RoadGraph(nodes, arcs), beta=BETA)
route = load_trip("data/student_test", tid)[3]
def arc_line(a): u, v = arcs.u[a], arcs.v[a]; return [(nodes.lat[u], nodes.lon[u]), (nodes.lat[v], nodes.lon[v])]
m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=15)
layers = {"Road network (near route)": "lightgray", "Reference route": "green", "Predicted route (HMM)": "red"}
fg = {k: folium.FeatureGroup(k) for k in layers}
near = set(feats.arc_id.unique())
for a in near: folium.PolyLine(arc_line(a), color="lightgray", weight=2).add_to(fg["Road network (near route)"])
for a in route: folium.PolyLine(arc_line(a), color="green", weight=7, opacity=0.6).add_to(fg["Reference route"])
for a in sorted(set(pred.values())): folium.PolyLine(arc_line(a), color="red", weight=3).add_to(fg["Predicted route (HMM)"])
gps = folium.FeatureGroup("Raw GPS points")
for la, lo in track[["lat", "lon"]].values[::3]: folium.CircleMarker((la, lo), radius=2, color="blue", fill=True).add_to(gps)
for g in list(fg.values()) + [gps]: g.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
os.makedirs("maps", exist_ok=True); m.save(f"maps/route_{tid}.html"); print(f"saved maps/route_{tid}.html")
