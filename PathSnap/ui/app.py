"""PathSnap evaluator UI (Laptop 2).  Run:  streamlit run ui/app.py
Talks ONLY to the FastAPI backend on Laptop 1. It never sees ground-truth routes."""
import os, time, requests, folium, pandas as pd, streamlit as st
import streamlit.components.v1 as components
import json

APP_NAME = "SnapRoute"                      # change the website name here
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
MAP_STYLES = {"Streets": ESRI.format("World_Street_Map"), "Dark": ESRI.format("Canvas/World_Dark_Gray_Base"),
              "Satellite": ESRI.format("World_Imagery")}
NICE = {"final_model": "HistGB tuned (final)", "HistGB": "HistGB (default)", "RandForest": "Random Forest",
        "MLP": "MLP neural net", "DecTree": "Decision Tree", "KNN": "KNN", "LogReg": "Logistic Regression",
        "GaussNB": "Gaussian Naive Bayes"}
nice = lambda m: NICE.get(m, m)
PCT = ["Top-1 accuracy", "Route continuity", "Arc precision", "Arc recall"]

st.set_page_config(page_title=f"{APP_NAME} · PathSnap", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {font-family: 'Inter', sans-serif;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
.ps-hero {position: relative; overflow: hidden; padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.1rem; color: #fff;
          background: linear-gradient(120deg, #0f172a, #1e3a8a, #0e7490, #0f766e); background-size: 300% 300%;
          animation: psflow 14s ease infinite; box-shadow: 0 10px 30px rgba(14,116,144,.25);}
@keyframes psflow {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
.ps-hero h1 {margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -.02em; color: #fff;}
.ps-hero h1 small {font-size: .95rem; font-weight: 600; opacity: .8; margin-left: .6rem; letter-spacing: 0;}
.ps-hero p {margin: .35rem 0 .9rem; opacity: .9; font-size: 1rem; max-width: 900px;}
.ps-chips {display: flex; flex-wrap: wrap; gap: .45rem; align-items: center;}
.ps-chip {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); padding: .28rem .7rem;
          border-radius: 999px; font-size: .82rem; font-weight: 600; backdrop-filter: blur(4px);}
.ps-arrow {opacity: .7; font-size: .9rem;}
.ps-stats {display: flex; gap: 1.6rem; margin-top: .9rem; font-size: .85rem; opacity: .95;}
.ps-stats b {font-size: 1.15rem; display: block;}
.ps-card {border-radius: 14px; padding: .95rem 1rem; height: 100%; border: 1px solid rgba(128,128,128,.22);
          border-left: 5px solid var(--acc, #0e7490); background: rgba(128,128,128,.07); transition: transform .15s, box-shadow .15s;}
.ps-card:hover {transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,.18);}
.ps-card .lbl {font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7; font-weight: 600;}
.ps-card .val {font-size: 1.85rem; font-weight: 800; margin-top: .1rem; color: var(--acc, inherit);}
.ps-card .hint {font-size: .78rem; opacity: .7; margin-top: .1rem;}
.ps-legend span {display: inline-block; margin-right: 1.2rem; font-size: .9rem;}
.ps-dot {display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle;}
.ps-line {display: inline-block; width: 22px; height: 4px; margin-right: 6px; vertical-align: middle; border-radius: 2px;}
.stTabs [data-baseweb="tab-list"] {gap: .4rem;}
.stTabs [data-baseweb="tab"] {padding: .5rem 1rem; border-radius: 10px 10px 0 0; font-weight: 600;}
div[data-testid="stButton"] button[kind="primary"] {border-radius: 10px; font-weight: 700;
          background: linear-gradient(90deg, #0e7490, #1e40af); border: 0;}
div[data-testid="stButton"] button[kind="primary"]:hover {filter: brightness(1.12);}
.ps-footer {margin-top: 2.2rem; padding-top: .8rem; border-top: 1px solid rgba(128,128,128,.25);
            font-size: .8rem; opacity: .65; text-align: center;}
</style>""", unsafe_allow_html=True)

for k, v in {"history": [], "result": None, "multi": None}.items():
    st.session_state.setdefault(k, v)


def quality(x):
    """green >= 90 %, amber >= 75 %, red below; grey when not a percentage"""
    if x is None:
        return "#64748b"
    return "#16a34a" if x >= .90 else "#d97706" if x >= .75 else "#dc2626"


def cards(items):
    for col, it in zip(st.columns(len(items)), items):
        lbl, val, hint = it[:3]; acc = it[3] if len(it) > 3 else "#0e7490"
        col.markdown(f'<div class="ps-card" style="--acc:{acc}"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
                     f'<div class="hint">{hint}</div></div>', unsafe_allow_html=True)


def pct_cols(cols):
    return {k: st.column_config.ProgressColumn(k, format="percent", min_value=0, max_value=1) for k in cols}


def call_predict(trip, model, use_hmm):
    r = requests.post(f"{api}/predict", json={"trip_id": trip, "model": model, "use_hmm": use_hmm}, timeout=900)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()


def row_from(out):
    m = out["metrics"]
    return {"Trip": out["trip_id"], "Model": nice(out["model"]), "Decoder": "HMM" if out["use_hmm"] else "Classifier only",
            "Top-1 accuracy": m.get("point_top1_accuracy"), "Route continuity": m.get("route_continuity"),
            "Arc precision": m.get("route_arc_precision"), "Arc recall": m.get("route_arc_recall"),
            "Inference (s)": m.get("inference_time_s"), "GPS points": m.get("n_gps_points")}


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Connection")
    api = st.text_input("API address (Laptop 1)", os.environ.get("API_URL", "http://127.0.0.1:8000")).rstrip("/")
    try:
        health = requests.get(f"{api}/health", timeout=5).json(); ok = health.get("status") == "ok"
    except Exception:
        health, ok = {}, False
    if ok:
        st.success(f"API connected · {health.get('models', '?')} models · HMM beta {health.get('beta', '?')}", icon="✅")
    else:
        st.error("API not reachable", icon="⛔")
        st.caption("Check that uvicorn is running on Laptop 1, both laptops are on the same network, "
                   "and the address above uses Laptop 1's IPv4 address. Then press F5.")
    st.divider()
    st.caption("All prediction and scoring happen on Laptop 1. Reference routes never leave the server.")

def hero(n_trips="–", n_models="–", beta="–"):
    st.markdown(f"""<div class="ps-hero"><h1>🛰️ {APP_NAME}<small>a PathSnap map-matching system</small></h1>
<p>Turns noisy GPS traces into the road route a vehicle actually drove, using a machine-learning candidate
classifier and HMM / Viterbi decoding over the road network.</p>
<div class="ps-chips"><span class="ps-chip">📡 Raw GPS</span><span class="ps-arrow">➜</span>
<span class="ps-chip">🛣️ Candidate roads ≤ 50 m</span><span class="ps-arrow">➜</span>
<span class="ps-chip">🤖 ML classifier</span><span class="ps-arrow">➜</span>
<span class="ps-chip">🔗 HMM / Viterbi</span><span class="ps-arrow">➜</span>
<span class="ps-chip">🗺️ Matched route</span></div>
<div class="ps-stats"><span><b>{n_trips}</b>evaluation trips</span><span><b>{n_models}</b>trained models</span>
<span><b>{beta}</b>HMM beta</span></div></div>""", unsafe_allow_html=True)


if not ok:
    hero()
    st.info("Waiting for the API. Fix the connection in the left panel, then press F5 to reload this page.")
    st.stop()
try:
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
    model_list = requests.get(f"{api}/models", timeout=10).json()["models"]
except Exception as e:
    st.error(f"Could not load trips/models from the API: {e}"); st.stop()
if not trips:
    st.warning("The API has no evaluation trips in its data/eval_test folder."); st.stop()
default_model = "final_model" if "final_model" in model_list else model_list[0]
hero(len(trips), len(model_list), f"{health.get('beta', 0):g}")

tab_run, tab_dev, tab_multi = st.tabs(["🗺️  Run a trip", "📊  Model comparison (development)", "🧪  Compare models on a trip"])

# ================================================================ TAB 1: run one trip
with tab_run:
    st.subheader("1 · Choose a trip and a model")
    c1, c2, c3, c4 = st.columns([2, 2, 1.2, 1.4], vertical_alignment="bottom")
    trip = c1.selectbox(f"{len(trips)} evaluation trips", trips, format_func=lambda t: f"Trip {t}", key="t1")
    model = c2.selectbox("Model", model_list, index=model_list.index(default_model), format_func=nice, key="m1")
    use_hmm = c3.toggle("HMM decoding", value=True, key="h1")
    run = c4.button("▶  Run map-matching", type="primary", width="stretch")

    if run:
        with st.status(f"Processing trip {trip} on Laptop 1…", expanded=True) as status:
            st.write(f"Model: {nice(model)} · decoder: {'HMM / Viterbi' if use_hmm else 'classifier only'}")
            t0 = time.time()
            try:
                out = call_predict(trip, model, use_hmm)
            except Exception as e:
                status.update(label="Request failed", state="error"); st.error(str(e)); st.stop()
            out["round_trip_s"] = round(time.time() - t0, 2)
            status.update(label=f"Trip {trip} finished in {out['round_trip_s']} s", state="complete", expanded=False)
        st.session_state.result = out
        key = (trip, model, use_hmm)
        st.session_state.history = [h for h in st.session_state.history
                                    if (h["Trip"], h["Model"], h["Decoder"]) != (key[0], nice(key[1]), "HMM" if key[2] else "Classifier only")]
        st.session_state.history.append(row_from(out))

    out = st.session_state.result
    if out is None:
        st.caption("Pick a trip and press **Run map-matching**. Results appear here.")
    else:
        m = out["metrics"]
        pct = lambda k: f"{m[k]*100:.1f}%" if m.get(k) is not None else "n/a"
        st.subheader(f"2 · Results for trip {out['trip_id']}")
        st.caption(f"{nice(out['model'])} · {'HMM / Viterbi decoding' if out['use_hmm'] else 'classifier only'}")
        cards([("Point accuracy (Top-1)", pct("point_top1_accuracy"), "GPS points matched to the correct road", quality(m.get("point_top1_accuracy"))),
               ("Route continuity", pct("route_continuity"), "consecutive predicted roads that connect", quality(m.get("route_continuity"))),
               ("Arc precision", pct("route_arc_precision"), "predicted roads that are on the true route", quality(m.get("route_arc_precision"))),
               ("Arc recall", pct("route_arc_recall"), "true-route roads that we recovered", quality(m.get("route_arc_recall"))),
               ("Inference time", f"{m.get('inference_time_s', 0)} s", f"{m.get('n_gps_points', 0)} GPS points", "#6366f1")])
        st.caption("Card colour: 🟢 ≥ 90 %  ·  🟠 75–90 %  ·  🔴 < 75 %")
        st.write("")
        st.subheader("📥 Download prediction reports")

        download_cols = st.columns(3)

        with download_cols[0]:
            st.download_button(
                "⬇️ Full prediction JSON",
                data=json.dumps(out, indent=2, default=str),
                file_name=f"pathsnap_{out['trip_id']}_report.json",
                mime="application/json",
                use_container_width=True
            )

        with download_cols[1]:
            metrics_df = pd.DataFrame([m])
            st.download_button(
                "⬇️ Metrics CSV",
                data=metrics_df.to_csv(index=False),
                file_name=f"pathsnap_{out['trip_id']}_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )

        with download_cols[2]:
            history_df = pd.DataFrame(st.session_state.history)
            st.download_button(
                "⬇️ Session history CSV",
                data=history_df.to_csv(index=False),
                file_name="pathsnap_session_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        t_map, t_hist, t_raw = st.tabs(["Map", "Runs this session", "Raw API response"])
        with t_map:
            cc1, cc2 = st.columns([1, 2], vertical_alignment="center")
            show_gps = cc1.toggle("Show raw GPS points", value=True)
            style = cc2.radio("Map style", list(MAP_STYLES), horizontal=True, label_visibility="collapsed")
            gps, lines = out["gps"], out["predicted_route"]
            import math
            la = [q[0] for q in gps]; lo = [q[1] for q in gps]
            clat, clon = (min(la) + max(la)) / 2, (min(lo) + max(lo)) / 2
            extent_m = max((max(la) - min(la)) * 111320, (max(lo) - min(lo)) * 111320 * math.cos(math.radians(clat)), 200)
            zoom = int(max(3, min(18, math.floor(math.log2(156543 * math.cos(math.radians(clat)) * 700 / extent_m)))))
            fmap = folium.Map(location=[clat, clon], zoom_start=zoom, tiles=MAP_STYLES[style], attr="Esri", control_scale=True)
            if show_gps:
                g = folium.FeatureGroup("Raw GPS")
                for p in gps:
                    folium.CircleMarker(p, radius=2, color="#2563eb", fill=True, fill_opacity=.8, weight=0).add_to(g)
                g.add_to(fmap)
            r_ = folium.FeatureGroup("Predicted route")
            for line in lines:
                folium.PolyLine(line, color="#ef4444" if style != "Streets" else "#dc2626", weight=5, opacity=.95).add_to(r_)
            r_.add_to(fmap)
            folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
            folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(fmap)
            folium.LayerControl(collapsed=True).add_to(fmap)
            lats, lons = [p[0] for p in gps], [p[1] for p in gps]
            st.markdown('<div class="ps-legend">'
                        '<span><span class="ps-dot" style="background:#2563eb"></span>Raw GPS</span>'
                        '<span><span class="ps-line" style="background:#dc2626"></span>Predicted route</span>'
                        '<span>🟢 Start &nbsp; ⚫ End</span></div>', unsafe_allow_html=True)
            html = fmap.get_root().render()
            if hasattr(st, "iframe"):
                st.iframe(html, height=560)
            else:
                components.html(html, height=560)
        with t_hist:
            st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, width="stretch", column_config=pct_cols(PCT))
        with t_raw:
            st.json(m)
            st.caption(f"Round-trip time seen by the UI: {out.get('round_trip_s', '?')} s")

# ================================================================ TAB 2: development results
with tab_dev:
    st.subheader("How the final model was chosen")
    st.caption("All numbers below come from the development phase: 5-fold trip-grouped cross-validation on the "
               "80 training trips and evaluation on the 10 student-test trips. Evaluator trips were never used.")
    try:
        dev = requests.get(f"{api}/dev_results", timeout=20).json()
    except Exception as e:
        dev = {}; st.error(f"Could not load development results: {e}")
    if "models" in dev:
        df = pd.DataFrame(dev["models"])
        best = df.loc[df.test_top1.idxmax()]
        cards([("Models compared", str(len(df)), f"{df.family.nunique()} algorithm families"),
               ("Best student-test Top-1", f"{best.test_top1*100:.1f}%", best.model),
               ("Best log loss", f"{df.test_log_loss.min():.3f}", df.loc[df.test_log_loss.idxmin(), 'model']),
               ("Fastest training", f"{df.train_time_s.min():.2f} s", df.loc[df.train_time_s.idxmin(), 'model'])])
        st.write("")
        show = df.rename(columns={"model": "Model", "family": "Family", "train_rows": "Train rows",
                                  "cv_macro_f1_mean": "CV macro F1", "cv_macro_f1_std": "CV std",
                                  "test_macro_f1": "Test macro F1", "test_precision": "Precision",
                                  "test_recall": "Recall", "test_log_loss": "Log loss", "test_top1": "Top-1",
                                  "train_time_s": "Train (s)", "inference_time_s": "Inference (s)"}).sort_values("Top-1", ascending=False)
        show = show[["Model", "Family", "Top-1", "Test macro F1", "CV macro F1", "CV std", "Log loss", "Precision",
                     "Recall", "Train (s)", "Inference (s)", "Train rows"]]
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config=pct_cols(["CV macro F1", "Test macro F1", "Precision", "Recall", "Top-1"]))
        metric = st.radio("Chart", ["Top-1", "Test macro F1", "CV macro F1", "Log loss", "Train (s)", "Inference (s)"],
                          horizontal=True)
        st.bar_chart(show.set_index("Model")[[metric]], horizontal=True, height=320)
        st.caption("Lower is better for log loss and times; higher is better for the rest.")
    else:
        st.info("No results/model_comparison.csv found on the API laptop.")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### Bayesian optimisation (Optuna TPE)")
        if "tuning" in dev:
            st.dataframe(pd.DataFrame(dev["tuning"]), hide_index=True, width="stretch")
        if "optuna_best" in dev:
            ob = dev["optuna_best"]
            st.caption(f"{ob.get('n_trials')} trials · {ob.get('search_time_s', 0):.0f} s · best CV macro F1 "
                       f"{ob.get('best_cv_macro_f1', 0):.4f}")
            st.json(ob.get("best_params", {}), expanded=False)
    with cB:
        st.markdown("#### HMM decoder: effect of beta")
        if "hmm" in dev:
            h = pd.DataFrame(dev["hmm"])
            st.dataframe(h, hide_index=True, width="stretch", column_config=pct_cols(["top1", "continuity"]))
            st.caption("Average over the 10 student-test trips. Small beta = stricter road connectivity.")

# ================================================================ TAB 3: all models on one trip
with tab_multi:
    st.subheader("Run several models on the same trip")
    st.caption("Shows how each saved model behaves on one evaluation trip. For display only: the final model was "
               "selected on the student-test trips, not on these.")
    c1, c2, c3 = st.columns([2, 3, 1.4], vertical_alignment="bottom")
    trip3 = c1.selectbox("Trip", trips, format_func=lambda t: f"Trip {t}", key="t3")
    chosen = c2.multiselect("Models", model_list, default=model_list, format_func=nice)
    both = st.checkbox("Also run each model without HMM (classifier only)", value=True)
    go = c3.button("▶  Compare", type="primary", width="stretch")
    if go and chosen:
        jobs = [(m, True) for m in chosen] + ([(m, False) for m in chosen] if both else [])
        bar = st.progress(0.0, text="Starting…"); rows = []
        for i, (m, hm) in enumerate(jobs, 1):
            bar.progress((i - 1) / len(jobs), text=f"{nice(m)} · {'HMM' if hm else 'classifier only'} ({i}/{len(jobs)})")
            try:
                rows.append(row_from(call_predict(trip3, m, hm)))
            except Exception as e:
                st.warning(f"{nice(m)} failed: {e}")
        bar.progress(1.0, text="Done")
        st.session_state.multi = pd.DataFrame(rows)
    res = st.session_state.multi
    if res is not None and len(res):
        res = res.sort_values("Top-1 accuracy", ascending=False)
        top = res.iloc[0]
        st.success(f"Best on trip {top['Trip']}: **{top['Model']}** with {top['Decoder']} "
                   f"({top['Top-1 accuracy']*100:.1f}% Top-1)")
        st.dataframe(res, hide_index=True, width="stretch", column_config=pct_cols(PCT))
        wide = res.pivot_table(index="Model", columns="Decoder", values="Top-1 accuracy")
        st.markdown("**Top-1 accuracy per model**")
        st.bar_chart(wide, horizontal=True, height=320, stack=False)
        if {"HMM", "Classifier only"} <= set(res.Decoder):
            cont = res.pivot_table(index="Model", columns="Decoder", values="Route continuity")
            st.markdown("**Route continuity per model** (what the HMM is designed to fix)")
            st.bar_chart(cont, horizontal=True, height=320, stack=False)

st.markdown(f'<div class="ps-footer">{APP_NAME} · PathSnap hackathon (UE24CS352A) · FastAPI backend on Laptop 1 · '
            'Streamlit UI on Laptop 2 · ground-truth routes stay server-side</div>', unsafe_allow_html=True)
