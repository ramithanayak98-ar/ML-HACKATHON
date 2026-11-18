"""PERSON D - Laptop 2: Streamlit evaluator UI.  Run:  streamlit run ui/app.py"""
import os, requests, folium, streamlit as st
import streamlit.components.v1 as components
API = st.sidebar.text_input("API address (Laptop 1)", os.environ.get("API_URL", "http://127.0.0.1:8000"))
st.title("PathSnap - Map-Matching & Route Reconstruction")
try:
    trips = requests.get(f"{API}/trips", timeout=10).json()["trips"]
except Exception as e:
    st.error(f"Cannot reach the API at {API}: {e}"); st.stop()
trip = st.radio("Choose one of the evaluation trips", trips, horizontal=True)
if st.button("Run map-matching"):
    with st.spinner("Sending trip to the API..."):
        r = requests.post(f"{API}/predict", json={"trip_id": trip}, timeout=600)
    if r.status_code != 200:
        st.error(r.text); st.stop()
    out = r.json(); m = out["metrics"]
    c = st.columns(4)
    pct = lambda k: f"{m[k]*100:.1f}%" if k in m else "n/a"
    c[0].metric("Point Top-1 accuracy", pct("point_top1_accuracy"))
    c[1].metric("Route continuity", pct("route_continuity"))
    c[2].metric("Arc recall", pct("route_arc_recall"))
    c[3].metric("Inference time", f"{m['inference_time_s']} s")
    gps = out["gps"]; fmap = folium.Map(location=gps[len(gps)//2], zoom_start=15)
    g1 = folium.FeatureGroup("Raw GPS"); g2 = folium.FeatureGroup("Predicted route")
    for p in gps: folium.CircleMarker(p, radius=2, color="blue").add_to(g1)
    for line in out["predicted_route"]: folium.PolyLine(line, color="red", weight=4).add_to(g2)
    g1.add_to(fmap); g2.add_to(fmap); folium.LayerControl().add_to(fmap)
    html = fmap.get_root().render()
    if hasattr(st, "iframe"): st.iframe(html, height=600)        # Streamlit >= 1.5x
    else: components.html(html, height=600)                     # older Streamlit
    st.json(m)
