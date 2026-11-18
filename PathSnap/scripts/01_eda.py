"""PERSON A - Step A3: dataset exploration. Run:  python scripts/01_eda.py"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import networkx as nx, folium
sys.path.insert(0, os.getcwd())
from src.core import load_trip
os.makedirs("results/eda", exist_ok=True); os.makedirs("maps", exist_ok=True)
rows = []
for part in ["train", "student_test"]:
    for tid in sorted(os.listdir(f"data/{part}")):
        tr, no, ar, ro = load_trip(f"data/{part}", tid)
        dt = np.diff(tr.t.values)
        G = nx.DiGraph(); G.add_nodes_from(range(len(no))); G.add_edges_from(ar[["u", "v"]].values)
        rows.append(dict(partition=part, trip_id=tid, n_gps=len(tr), n_nodes=len(no), n_arcs=len(ar),
            n_route_arcs=0 if ro is None else len(ro), missing_values=int(tr.isna().sum().sum()),
            duration_s=tr.t.iloc[-1] - tr.t.iloc[0], pct_1s_steps=float(np.mean(dt == 1) * 100),
            n_gaps_over_5s=int((dt > 5).sum()), max_gap_s=float(dt.max()), timestamps_sorted=bool((dt > 0).all()),
            lon_min=tr.lon.min(), lon_max=tr.lon.max(), lat_min=tr.lat.min(), lat_max=tr.lat.max(),
            weak_components=nx.number_weakly_connected_components(G),
            isolated_nodes=int(sum(1 for n in G.nodes if G.degree(n) == 0))))
        print(part, tid, "done")
inv = pd.DataFrame(rows); inv.to_csv("results/eda/dataset_inventory.csv", index=False)
print(inv.groupby("partition")[["n_gps", "n_arcs", "n_gaps_over_5s", "max_gap_s"]].describe().T)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
inv.n_gps.hist(ax=ax[0], bins=30); ax[0].set_title("GPS points per trip")
inv.n_arcs.hist(ax=ax[1], bins=30); ax[1].set_title("Road arcs per trip")
inv.max_gap_s.hist(ax=ax[2], bins=30); ax[2].set_title("Largest time gap per trip (s)")
plt.tight_layout(); plt.savefig("results/eda/histograms.png", dpi=150)
# Folium map of one training trip: GPS (blue), road network (grey), reference route (green)
tid = sorted(os.listdir("data/train"))[0]
tr, no, ar, ro = load_trip("data/train", tid)
m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15)
net = folium.FeatureGroup("Road network (sample)")
for u, v in ar.sample(min(3000, len(ar)), random_state=0).values:
    folium.PolyLine([(no.lat[u], no.lon[u]), (no.lat[v], no.lon[v])], color="gray", weight=1).add_to(net)
net.add_to(m)
ref = folium.FeatureGroup("Reference route")
for a in ro:
    u, v = ar.u[a], ar.v[a]
    folium.PolyLine([(no.lat[u], no.lon[u]), (no.lat[v], no.lon[v])], color="green", weight=5).add_to(ref)
ref.add_to(m)
gps = folium.FeatureGroup("Raw GPS")
for la, lo in tr[["lat", "lon"]].values[::5]:
    folium.CircleMarker((la, lo), radius=2, color="blue").add_to(gps)
gps.add_to(m); folium.LayerControl().add_to(m)
m.save(f"maps/eda_trip_{tid}.html"); print("saved map")
