"""PathSnap evaluator UI (Laptop 2).  Run:  streamlit run ui/app.py
Talks ONLY to the FastAPI backend on Laptop 1. It never sees ground-truth routes.
Person D - Deployment Lead UI Polish
"""
import os, time, math, requests, folium, pandas as pd, streamlit as st
import streamlit.components.v1 as components

APP_NAME = "SnapRoute"
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
MAP_STYLES = {
    "Streets": ESRI.format("World_Street_Map"),
    "Dark Canvas": ESRI.format("Canvas/World_Dark_Gray_Base"),
    "Satellite Imagery": ESRI.format("World_Imagery")
}
NICE = {
    "final_model": "HistGB Tuned (Final Model)",
    "HistGB": "HistGB (Baseline Default)",
    "RandForest": "Random Forest",
    "MLP": "MLP Neural Network",
    "DecTree": "Decision Tree",
    "KNN": "K-Nearest Neighbors",
    "LogReg": "Logistic Regression",
    "GaussNB": "Gaussian Naive Bayes"
}
nice = lambda m: NICE.get(m, m)
PCT = ["Top-1 accuracy", "Route continuity", "Arc precision", "Arc recall"]

st.set_page_config(
    page_title=f"{APP_NAME} · PathSnap Map-Matching",
    page_icon="🛰️",
    layout="wide"
)

# Enhanced Styling & Typography
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1400px; }

/* Hero Banner */
.ps-hero {
    position: relative;
    overflow: hidden;
    padding: 1.8rem 2.0rem;
    border-radius: 18px;
    margin-bottom: 1.3rem;
    color: #ffffff;
    background: linear-gradient(135deg, #091e3a 0%, #173868 35%, #0e7490 75%, #0f766e 100%);
    box-shadow: 0 10px 32px rgba(14, 116, 144, 0.22);
    border: 1px solid rgba(255, 255, 255, 0.12);
}
.ps-hero h1 { margin: 0; font-size: 2.35rem; font-weight: 800; letter-spacing: -0.025em; color: #ffffff; }
.ps-hero h1 small { font-size: 1.0rem; font-weight: 500; opacity: 0.85; margin-left: 0.6rem; letter-spacing: 0; }
.ps-hero p { margin: 0.45rem 0 1.0rem; opacity: 0.92; font-size: 1.02rem; max-width: 960px; line-height: 1.5; }
.ps-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
.ps-chip {
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.28);
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    backdrop-filter: blur(6px);
}
.ps-arrow { opacity: 0.75; font-size: 0.95rem; }
.ps-stats { display: flex; gap: 2.2rem; margin-top: 1.1rem; font-size: 0.88rem; opacity: 0.95; border-top: 1px solid rgba(255,255,255,0.18); padding-top: 0.85rem; }
.ps-stats b { font-size: 1.25rem; display: block; font-weight: 700; }

/* Metric Cards */
.ps-card {
    border-radius: 14px;
    padding: 1.05rem 1.15rem;
    height: 100%;
    border: 1px solid rgba(128, 128, 128, 0.24);
    border-left: 5px solid var(--acc, #0e7490);
    background: rgba(128, 128, 128, 0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.ps-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15); }
.ps-card .lbl { font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.07em; opacity: 0.75; font-weight: 700; }
.ps-card .val { font-size: 1.95rem; font-weight: 800; margin-top: 0.2rem; color: var(--acc, inherit); letter-spacing: -0.02em; }
.ps-card .hint { font-size: 0.80rem; opacity: 0.75; margin-top: 0.15rem; }

/* Map Legend */
.ps-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1.2rem;
    padding: 0.65rem 0.95rem;
    border-radius: 10px;
    background: rgba(128, 128, 128, 0.07);
    margin-bottom: 0.75rem;
    font-size: 0.88rem;
    border: 1px solid rgba(128, 128, 128, 0.18);
}
.ps-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.ps-line { display: inline-block; width: 22px; height: 5px; margin-right: 6px; vertical-align: middle; border-radius: 2px; }

/* Buttons & Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; margin-bottom: 0.8rem; }
.stTabs [data-baseweb="tab"] { padding: 0.6rem 1.15rem; border-radius: 10px 10px 0 0; font-weight: 600; font-size: 0.94rem; }
div[data-testid="stButton"] button[kind="primary"] {
    border-radius: 10px;
    font-weight: 700;
    background: linear-gradient(90deg, #0e7490, #1e40af);
    border: 0;
    padding: 0.55rem 1.25rem;
    box-shadow: 0 4px 14px rgba(14, 116, 144, 0.35);
}
div[data-testid="stButton"] button[kind="primary"]:hover { filter: brightness(1.14); }

/* Footer */
.ps-footer {
    margin-top: 2.8rem;
    padding-top: 1.1rem;
    border-top: 1px solid rgba(128, 128, 128, 0.22);
    font-size: 0.82rem;
    opacity: 0.72;
    text-align: center;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
for k, v in {"history": [], "result": None, "multi": None}.items():
    st.session_state.setdefault(k, v)


def quality(x):
    """Green >= 90%, Amber >= 75%, Red below; Slate when not a percentage"""
    if x is None:
        return "#64748b"
    return "#10b981" if x >= 0.90 else "#f59e0b" if x >= 0.75 else "#ef4444"


def cards(items):
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        lbl, val, hint = it[:3]
        acc = it[3] if len(it) > 3 else "#0e7490"
        col.markdown(
            f'<div class="ps-card" style="--acc:{acc}">'
            f'<div class="lbl">{lbl}</div>'
            f'<div class="val">{val}</div>'
            f'<div class="hint">{hint}</div></div>',
            unsafe_allow_html=True
        )


def pct_cols(cols):
    return {k: st.column_config.ProgressColumn(k, format="percent", min_value=0, max_value=1) for k in cols}


def call_predict(trip, model, use_hmm):
    r = requests.post(f"{api}/predict", json={"trip_id": trip, "model": model, "use_hmm": use_hmm}, timeout=900)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()


def row_from(out):
    m = out["metrics"]
    return {
        "Trip": out["trip_id"],
        "Model": nice(out["model"]),
        "Decoder": "HMM / Viterbi" if out["use_hmm"] else "Classifier only",
        "Top-1 accuracy": m.get("point_top1_accuracy"),
        "Route continuity": m.get("route_continuity"),
        "Arc precision": m.get("route_arc_precision"),
        "Arc recall": m.get("route_arc_recall"),
        "Inference (s)": m.get("inference_time_s"),
        "GPS points": m.get("n_gps_points")
    }


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ System Connection")
    default_api = os.environ.get("API_URL", "http://127.0.0.1:8000")
    api = st.text_input("FastAPI Backend URL (Laptop 1):", default_api).rstrip("/")

    # Quick Localhost helper
    if api != "http://127.0.0.1:8000" and st.button("🔄 Reset to Localhost (127.0.0.1:8000)", use_container_width=True):
        st.info("To test on the same machine, clear the text box and enter `http://127.0.0.1:8000`.")

    try:
        health = requests.get(f"{api}/health", timeout=4).json()
        ok = health.get("status") == "ok"
    except Exception:
        health, ok = {}, False

    if ok:
        st.success(f"🟢 Connected to Laptop 1\n\n• Models: **{health.get('models', '?')}**\n• HMM Beta: **{health.get('beta', '?')}**", icon="✅")
    else:
        st.error("🔴 Backend Server Unreachable", icon="⛔")
        st.markdown(
            """
            **Troubleshooting Steps:**
            1. **Same laptop test**: Ensure URL is `http://127.0.0.1:8000`.
            2. **Two-laptop test**: Ensure URL matches Laptop 1's Wi-Fi IP (e.g. `http://10.x.x.x:8000`).
            3. Confirm `uvicorn` is actively running on Laptop 1:
               ```cmd
               uvicorn api.main:app --host 0.0.0.0 --port 8000
               ```
            4. Confirm port 8000 is open in Windows Firewall.
            """
        )

    st.markdown("---")
    st.markdown(
        "**🔒 Privacy & Data Protection**\n\n"
        "All candidate generation, feature assembly, ML inference, and Viterbi decoding execute strictly on **Laptop 1**.\n\n"
        "Ground-truth reference routes never leave the server."
    )


def hero(n_trips="–", n_models="–", beta="–"):
    st.markdown(f"""
    <div class="ps-hero">
        <h1>🛰️ {APP_NAME}<small>PathSnap Map-Matching & Route Reconstruction</small></h1>
        <p>Reconstructs the true sequence of road segments traversed by vehicles from sparse, noisy GPS coordinates using KD-Tree spatial indexing, supervised machine learning probability scoring, and topological HMM / Viterbi sequence decoding.</p>
        <div class="ps-chips">
            <span class="ps-chip">📡 Raw GPS Observations</span><span class="ps-arrow">➜</span>
            <span class="ps-chip">🛣️ Spatial Candidate Arcs (r ≤ 50 m)</span><span class="ps-arrow">➜</span>
            <span class="ps-chip">🤖 ML Classifier Probabilities</span><span class="ps-arrow">➜</span>
            <span class="ps-chip">🔗 HMM Viterbi Decoder</span><span class="ps-arrow">➜</span>
            <span class="ps-chip">🗺️ Reconstructed Route</span>
        </div>
        <div class="ps-stats">
            <span><b>{n_trips}</b>Evaluation Journeys</span>
            <span><b>{n_models}</b>Trained Models Available</span>
            <span><b>{beta}</b>HMM Spatial Beta (m)</span>
            <span><b>UE24CS352A</b>ML Hackathon</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if not ok:
    hero()
    st.info("👈 Enter the active FastAPI Backend URL in the left sidebar and verify connection to begin.")
    st.stop()

try:
    trips = requests.get(f"{api}/trips", timeout=10).json().get("trips", [])
    model_list = requests.get(f"{api}/models", timeout=10).json().get("models", [])
except Exception as e:
    st.error(f"Failed to fetch metadata from FastAPI backend: {e}")
    st.stop()

if not trips:
    st.warning("No evaluation journeys found in `data/eval_test/` on Laptop 1.")
    st.stop()

default_model = "final_model" if "final_model" in model_list else model_list[0]
hero(len(trips), len(model_list), f"{health.get('beta', 0):g}")

tab_run, tab_dev, tab_multi = st.tabs([
    "🗺️  Run Route Reconstruction",
    "📊  Model Comparison & Benchmarks",
    "🧪  Multi-Model Evaluation on Trip"
])

# ================================================================ TAB 1: RUN ONE TRIP
with tab_run:
    st.subheader("1 · Configure Journey & Model")
    c1, c2, c3, c4 = st.columns([2.2, 2.2, 1.4, 1.6], vertical_alignment="bottom")
    trip = c1.selectbox(f"Select Evaluation Journey ({len(trips)} available):", trips, format_func=lambda t: f"Journey {t}", key="t1")
    model = c2.selectbox("Model Architecture:", model_list, index=model_list.index(default_model), format_func=nice, key="m1")
    use_hmm = c3.toggle("Apply HMM Decoder", value=True, key="h1", help="When disabled, greedy candidate selection (classifier-only) is performed.")
    run = c4.button("🚀 Reconstruct Route", type="primary", use_container_width=True)

    if run:
        with st.status(f"Processing Journey {trip} on Laptop 1…", expanded=True) as status:
            st.write(f"• Model: **{nice(model)}**")
            st.write(f"• Decoder: **{'HMM / Viterbi Sequence Decoder' if use_hmm else 'Classifier-Only Greedy Selection'}**")
            t0 = time.time()
            try:
                out = call_predict(trip, model, use_hmm)
            except Exception as e:
                status.update(label="Inference request failed", state="error")
                st.error(f"Prediction Error: {e}")
                st.stop()
            out["round_trip_s"] = round(time.time() - t0, 2)
            status.update(label=f"Journey {trip} map-matched in {out['round_trip_s']} s", state="complete", expanded=False)

        st.session_state.result = out
        key = (trip, model, use_hmm)
        st.session_state.history = [
            h for h in st.session_state.history
            if (h["Trip"], h["Model"], h["Decoder"]) != (key[0], nice(key[1]), "HMM / Viterbi" if key[2] else "Classifier only")
        ]
        st.session_state.history.append(row_from(out))

    out = st.session_state.result
    if out is None:
        st.info("Select an evaluation journey above and click **Reconstruct Route** to view metrics and interactive visualization.")
    else:
        m = out["metrics"]
        pct = lambda k: f"{m[k]*100:.1f}%" if m.get(k) is not None else "N/A"
        st.subheader(f"2 · Evaluation Metrics for Journey {out['trip_id']}")
        st.caption(f"Evaluated with **{nice(out['model'])}** · **{'HMM / Viterbi Decoding' if out['use_hmm'] else 'Classifier-Only Decoding'}**")

        cards([
            ("Point Accuracy (Top-1)", pct("point_top1_accuracy"), "Fraction of GPS points assigned to true road arc", quality(m.get("point_top1_accuracy"))),
            ("Route Continuity", pct("route_continuity"), "Fraction of consecutive road segments connected", quality(m.get("route_continuity"))),
            ("Arc Precision", pct("route_arc_precision"), "Predicted road segments present on true route", quality(m.get("route_arc_precision"))),
            ("Arc Recall", pct("route_arc_recall"), "Ground-truth road segments recovered", quality(m.get("route_arc_recall"))),
            ("Inference Time", f"{m.get('inference_time_s', 0)} s", f"{m.get('n_gps_points', 0)} GPS points processed", "#6366f1")
        ])
        st.caption("Metric Thresholds: 🟢 ≥ 90% (Strong)  ·  🟠 75%–89% (Moderate)  ·  🔴 < 75% (Sub-optimal)")
        st.write("")

        t_map, t_hist, t_raw = st.tabs(["🗺️ Interactive Map", "📜 Session Evaluation History", "📋 Raw API Response"])

        with t_map:
            cc1, cc2 = st.columns([1.2, 2], vertical_alignment="center")
            show_gps = cc1.toggle("Show GPS Observation Points", value=True)
            style = cc2.radio("Basemap Style", list(MAP_STYLES.keys()), horizontal=True, label_visibility="collapsed")
            gps, lines = out["gps"], out["predicted_route"]

            if gps:
                la = [q[0] for q in gps]
                lo = [q[1] for q in gps]
                clat, clon = (min(la) + max(la)) / 2, (min(lo) + max(lo)) / 2
                extent_m = max((max(la) - min(la)) * 111320, (max(lo) - min(lo)) * 111320 * math.cos(math.radians(clat)), 200)
                zoom = int(max(3, min(18, math.floor(math.log2(156543 * math.cos(math.radians(clat)) * 700 / extent_m)))))

                fmap = folium.Map(
                    location=[clat, clon],
                    zoom_start=zoom,
                    tiles=MAP_STYLES[style],
                    attr="Esri",
                    control_scale=True
                )

                if show_gps:
                    g = folium.FeatureGroup(name="Raw GPS Observations (Subsampled)")
                    for p in gps:
                        folium.CircleMarker(
                            p,
                            radius=3,
                            color="#2563eb",
                            fill=True,
                            fill_color="#3b82f6",
                            fill_opacity=0.85,
                            weight=0,
                            tooltip="Raw GPS Observation"
                        ).add_to(g)
                    g.add_to(fmap)

                r_ = folium.FeatureGroup(name="Predicted Reconstructed Route")
                route_color = "#ef4444" if style != "Streets" else "#dc2626"
                for line in lines:
                    folium.PolyLine(
                        line,
                        color=route_color,
                        weight=5,
                        opacity=0.95,
                        tooltip="Reconstructed Road Arc"
                    ).add_to(r_)
                r_.add_to(fmap)

                # Start and End Markers
                folium.Marker(
                    gps[0],
                    tooltip="Origin (Start GPS)",
                    icon=folium.Icon(color="green", icon="play", prefix="fa")
                ).add_to(fmap)
                folium.Marker(
                    gps[-1],
                    tooltip="Destination (End GPS)",
                    icon=folium.Icon(color="red", icon="flag", prefix="fa")
                ).add_to(fmap)

                folium.LayerControl(collapsed=True).add_to(fmap)

                st.markdown(
                    '<div class="ps-legend">'
                    '<span><span class="ps-dot" style="background:#2563eb"></span><b>Raw GPS Points</b></span>'
                    '<span><span class="ps-line" style="background:#dc2626"></span><b>Predicted Reconstructed Route</b></span>'
                    '<span>🟢 <b>Start Point</b></span>'
                    '<span>🚩 <b>End Point</b></span>'
                    '</div>',
                    unsafe_allow_html=True
                )

                html = fmap.get_root().render()
                if hasattr(st, "iframe"):
                    st.iframe(html, height=580)
                else:
                    components.html(html, height=580)
            else:
                st.warning("No GPS coordinates returned for this trip.")

        with t_hist:
            if st.session_state.history:
                st.dataframe(
                    pd.DataFrame(st.session_state.history),
                    hide_index=True,
                    use_container_width=True,
                    column_config=pct_cols(PCT)
                )
            else:
                st.caption("No history recorded yet in this session.")

        with t_raw:
            st.json(m)
            st.caption(f"Total round-trip request time measured by frontend: {out.get('round_trip_s', '?')} s")

# ================================================================ TAB 2: DEVELOPMENT RESULTS
with tab_dev:
    st.subheader("Development Benchmarks & Model Selection")
    st.caption(
        "All development benchmarks below derive from 5-fold GroupKFold cross-validation on the 80 training journeys "
        "and blind evaluation on the 10 student-test journeys. Evaluator trips were strictly isolated."
    )
    try:
        dev = requests.get(f"{api}/dev_results", timeout=20).json()
    except Exception as e:
        dev = {}
        st.error(f"Could not load development results from API: {e}")

    if "models" in dev and dev["models"]:
        df = pd.DataFrame(dev["models"])
        best = df.loc[df.test_top1.idxmax()]
        cards([
            ("Architectures Compared", str(len(df)), f"{df.family.nunique()} distinct model families"),
            ("Best Student-Test Top-1", f"{best.test_top1*100:.1f}%", best.model, "#10b981"),
            ("Lowest Log Loss", f"{df.test_log_loss.min():.3f}", df.loc[df.test_log_loss.idxmin(), 'model'], "#0e7490"),
            ("Fastest Training", f"{df.train_time_s.min():.2f} s", df.loc[df.train_time_s.idxmin(), 'model'], "#6366f1")
        ])
        st.write("")
        show = df.rename(columns={
            "model": "Model", "family": "Family", "train_rows": "Train Rows",
            "cv_macro_f1_mean": "CV Macro F1", "cv_macro_f1_std": "CV Std",
            "test_macro_f1": "Test Macro F1", "test_precision": "Precision",
            "test_recall": "Recall", "test_log_loss": "Log Loss", "test_top1": "Top-1",
            "train_time_s": "Train Time (s)", "inference_time_s": "Inference Time (s)"
        }).sort_values("Top-1", ascending=False)

        show = show[["Model", "Family", "Top-1", "Test Macro F1", "CV Macro F1", "CV Std", "Log Loss", "Precision", "Recall", "Train Time (s)", "Inference Time (s)", "Train Rows"]]
        st.dataframe(
            show,
            hide_index=True,
            use_container_width=True,
            column_config=pct_cols(["CV Macro F1", "Test Macro F1", "Precision", "Recall", "Top-1"])
        )
        metric = st.radio("Select Metric for Comparative Visualisation:", ["Top-1", "Test Macro F1", "CV Macro F1", "Log Loss", "Train Time (s)", "Inference Time (s)"], horizontal=True)
        st.bar_chart(show.set_index("Model")[[metric]], horizontal=True, height=320)
        st.caption("Note: Lower values are superior for Log Loss and Inference Time; higher values are superior for all accuracy metrics.")
    else:
        st.info("No development benchmarks (`results/model_comparison.csv`) loaded from backend.")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### 🔍 Bayesian Hyperparameter Tuning (Optuna TPE)")
        if "tuning" in dev:
            st.dataframe(pd.DataFrame(dev["tuning"]), hide_index=True, use_container_width=True)
        if "optuna_best" in dev:
            ob = dev["optuna_best"]
            st.caption(f"{ob.get('n_trials')} trials · Search time: {ob.get('search_time_s', 0):.0f} s · Best CV Macro F1: {ob.get('best_cv_macro_f1', 0):.4f}")
            st.json(ob.get("best_params", {}), expanded=False)

    with cB:
        st.markdown("#### 🔗 HMM Decoder Sensitivity to Beta Parameter")
        if "hmm" in dev:
            h = pd.DataFrame(dev["hmm"])
            st.dataframe(h, hide_index=True, use_container_width=True, column_config=pct_cols(["top1", "continuity"]))
            st.caption("Averages evaluated across all 10 student-test journeys. Stricter beta penalizes road detachment.")

# ================================================================ TAB 3: COMPARE ALL MODELS
with tab_multi:
    st.subheader("Multi-Model Comparative Evaluation on Single Journey")
    st.caption("Execute multiple candidate architectures on one evaluation journey to examine model stability and quantify the exact impact of HMM decoding.")
    c1, c2, c3 = st.columns([2.2, 3.2, 1.6], vertical_alignment="bottom")
    trip3 = c1.selectbox("Select Journey:", trips, format_func=lambda t: f"Journey {t}", key="t3")
    chosen = c2.multiselect("Select Models to Benchmark:", model_list, default=model_list, format_func=nice)
    both = st.checkbox("Include classifier-only run without HMM (ablation comparison)", value=True)
    go = c3.button("⚡ Compare Models", type="primary", use_container_width=True)

    if go and chosen:
        jobs = [(m, True) for m in chosen] + ([(m, False) for m in chosen] if both else [])
        bar = st.progress(0.0, text="Benchmarking in progress…")
        rows = []
        for i, (m, hm) in enumerate(jobs, 1):
            bar.progress((i - 1) / len(jobs), text=f"Evaluating {nice(m)} · {'HMM / Viterbi' if hm else 'Classifier-Only'} ({i}/{len(jobs)})")
            try:
                rows.append(row_from(call_predict(trip3, m, hm)))
            except Exception as e:
                st.warning(f"{nice(m)} benchmark failed: {e}")
        bar.progress(1.0, text="Benchmark complete!")
        st.session_state.multi = pd.DataFrame(rows)

    res = st.session_state.multi
    if res is not None and len(res):
        res = res.sort_values("Top-1 accuracy", ascending=False)
        top = res.iloc[0]
        st.success(f"Best Configuration on Journey {top['Trip']}: **{top['Model']}** with {top['Decoder']} (**{top['Top-1 accuracy']*100:.1f}% Top-1 Accuracy**)")
        st.dataframe(res, hide_index=True, use_container_width=True, column_config=pct_cols(PCT))

        wide = res.pivot_table(index="Model", columns="Decoder", values="Top-1 accuracy")
        st.markdown("**Top-1 Accuracy Across Architectures**")
        st.bar_chart(wide, horizontal=True, height=320, stack=False)

        if {"HMM / Viterbi", "Classifier only"} <= set(res.Decoder):
            cont = res.pivot_table(index="Model", columns="Decoder", values="Route continuity")
            st.markdown("**Route Continuity Improvement from HMM Viterbi Decoding**")
            st.bar_chart(cont, horizontal=True, height=320, stack=False)

st.markdown(
    f'<div class="ps-footer">'
    f'<b>{APP_NAME}</b> · PathSnap Map-Matching System · UE24CS352A Machine Learning Hackathon<br>'
    f'Distributed Architecture: FastAPI Backend (Laptop 1) · Streamlit Evaluation UI (Laptop 2)<br>'
    f'Data Privacy Guaranteed: Ground-truth reference routes remain strictly server-side.'
    f'</div>',
    unsafe_allow_html=True
)
