import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RDX20 AI & Telemetry Dashboard",
    layout="wide",
    page_icon="⚙️"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark industrial theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #0a0e1a; color: #e2e8f0; }

    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .block-container { padding-top: 1.5rem !important; }

    h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

    /* KPI cards */
    .kpi-card {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #00d4ff, transparent);
    }
    .kpi-card.orange::before { background: linear-gradient(90deg, #ff6b35, transparent); }
    .kpi-card.green::before  { background: linear-gradient(90deg, #7fff6b, transparent); }
    .kpi-card.red::before    { background: linear-gradient(90deg, #ff4b4b, transparent); }

    .kpi-label { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 2px;
                 text-transform: uppercase; color: #64748b; margin-bottom: 6px; }
    .kpi-value { font-family: 'Space Mono', monospace; font-size: 28px; font-weight: 700; color: #00d4ff; }
    .kpi-value.orange { color: #ff6b35; }
    .kpi-value.green  { color: #7fff6b; }
    .kpi-value.red    { color: #ff4b4b; }
    .kpi-sub { font-size: 11px; color: #64748b; margin-top: 4px; }

    /* Section header */
    .section-head {
        font-family: 'Space Mono', monospace;
        font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
        color: #00d4ff; padding: 6px 0 12px; border-bottom: 1px solid #1e3a5f;
        margin-bottom: 16px;
    }

    /* Metric comparison card */
    .metric-box {
        background: #111827; border: 1px solid #1e3a5f;
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .metric-box .m-label { font-size: 10px; font-family: 'Space Mono', monospace;
                           text-transform: uppercase; letter-spacing: 1px; color: #64748b; }
    .metric-box .m-val   { font-family: 'Space Mono', monospace; font-size: 22px;
                           font-weight: 700; margin-top: 4px; }

    /* Ra quality badge */
    .ra-good { color: #7fff6b; font-weight: 700; }
    .ra-warn { color: #ffd700; font-weight: 700; }
    .ra-bad  { color: #ff4b4b; font-weight: 700; }

    /* Prediction result */
    .pred-result {
        background: #111827; border: 1px solid #1e3a5f;
        border-radius: 12px; padding: 24px; text-align: center;
    }
    .pred-result .p-label { font-family: 'Space Mono', monospace; font-size: 11px;
                            letter-spacing: 2px; text-transform: uppercase; color: #64748b; }
    .pred-result .p-val   { font-family: 'Space Mono', monospace; font-size: 48px;
                            font-weight: 700; color: #7fff6b;
                            text-shadow: 0 0 20px rgba(127,255,107,0.4); }
    .pred-result .p-unit  { color: #64748b; font-size: 13px; margin-top: 4px; }

    div[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 10px; }

    .stSlider > div > div > div { background: #00d4ff !important; }
    .stSelectbox > div { background: #111827 !important; border: 1px solid #1e3a5f !important; }

    .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #64748b;
        font-family: 'Space Mono', monospace; font-size: 11px;
        letter-spacing: 1px; text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #00d4ff !important; }

    hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. FILE UPLOAD — SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Data Files")
    telemetry_file = st.file_uploader(
        "Telemetry CSV (signal_analys_*.csv)",
        type=["csv"],
        help="RDX20 signal analysis CSV export"
    )
    alarms_file = st.file_uploader(
        "Alarm Log CSV",
        type=["csv"],
        help="alarm_info_*.csv"
    )

if telemetry_file is None or alarms_file is None:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:70vh;text-align:center;'>
      <div style='font-size:56px;margin-bottom:24px;'>⚙️</div>
      <h2 style='font-family:Space Mono,monospace;color:#00d4ff;margin-bottom:12px;'>
        RDX20 AI Dashboard
      </h2>
      <p style='color:#64748b;font-size:14px;max-width:420px;line-height:1.7;'>
        Upload your two data files in the <strong style='color:#e2e8f0'>sidebar on the left</strong>
        to launch the dashboard.
      </p>
      <div style='margin-top:32px;display:flex;gap:20px;justify-content:center;flex-wrap:wrap;'>
        <div style='background:#111827;border:1px solid #1e3a5f;border-radius:10px;
                    padding:20px 28px;min-width:180px;'>
          <div style='font-family:Space Mono,monospace;font-size:10px;letter-spacing:2px;
                      text-transform:uppercase;color:#64748b;margin-bottom:8px;'>File 1</div>
          <div style='color:#00d4ff;font-weight:600;'>Telemetry CSV</div>
          <div style='color:#64748b;font-size:12px;margin-top:4px;'>signal_analys_*.csv</div>
          <div style='margin-top:10px;font-size:20px;'>{'✅' if telemetry_file else '⬜'}</div>
        </div>
        <div style='background:#111827;border:1px solid #1e3a5f;border-radius:10px;
                    padding:20px 28px;min-width:180px;'>
          <div style='font-family:Space Mono,monospace;font-size:10px;letter-spacing:2px;
                      text-transform:uppercase;color:#64748b;margin-bottom:8px;'>File 2</div>
          <div style='color:#ff6b35;font-weight:600;'>Alarm Log CSV</div>
          <div style='color:#64748b;font-size:12px;margin-top:4px;'>alarm_info_*.csv</div>
          <div style='margin-top:10px;font-size:20px;'>{'✅' if alarms_file else '⬜'}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

@st.cache_data
def load_telemetry(file_bytes):
    raw = pd.read_csv(file_bytes, encoding='utf-8-sig')
    signal_map = {
        "Feed rate F [actual]": "Feed_Rate",
        "Speed S [actual]":     "Speed",
        "Servo load current":   "Servo_Load",
        "Cutting feed signal":  "Cutting_Signal",
        "Spindle motor load":   "Spindle_Load",
    }
    df = pd.DataFrame()
    df["Time"] = raw["date"].astype(str)
    for keyword, col_name in signal_map.items():
        match = [c for c in raw.columns if keyword in c and c.endswith("-Value")]
        if not match:
            raise ValueError(f"Column not found for signal: '{keyword}'")
        df[col_name] = pd.to_numeric(raw[match[0]].replace("null", 0), errors="coerce").fillna(0)
    return df

@st.cache_data
def load_alarms(file_bytes):
    return pd.read_csv(file_bytes)

df_telemetry = load_telemetry(telemetry_file)
df_alarms    = load_alarms(alarms_file)


# ─────────────────────────────────────────────
# 3. SYNTHETIC RA DATASET (1000 samples)
# Coefficients fitted from actual L9 experimental
# data (aluminium CNC milling, Ra in µm):
#   K=0.0396, a=+0.6776, b=-0.2236, c=+0.7112
# Noise: multiplicative Gaussian, σ=40%
# (matches measured coefficient of variation
#  across all 9 experimental runs)
# Ra range: 0.08 – 0.87 µm (clipped to actual
#  observed min/max from experiments)
# ─────────────────────────────────────────────
@st.cache_data
def generate_ra_dataset(n=1000, seed=42):
    """
    Generates synthetic surface roughness dataset for aluminium CNC milling.
    Power-law coefficients are fitted from actual L9 experimental measurements:
      Ra = 0.0396 × feed^0.6776 × speed^-0.2236 × doc^0.7112
    Noise: multiplicative Gaussian (σ=0.40) matching experimental variability.
    """
    rng = np.random.default_rng(seed)

    # Parameter ranges match the L9 Taguchi levels used in the experiment
    spindle_speed = rng.uniform(400, 800, n)   # rpm
    feed_rate     = rng.uniform(100, 260, n)   # mm/min
    depth_of_cut  = rng.uniform(0.3,  1.2, n)  # mm

    # Power-law model with coefficients fitted from actual experimental data
    K, a, b, c = 0.0396, 0.6776, -0.2236, 0.7112
    Ra = K * (feed_rate ** a) * (spindle_speed ** b) * (depth_of_cut ** c)

    # Multiplicative Gaussian noise — σ=0.40 matches the ~40% coefficient
    # of variation observed across all 9 experimental runs
    noise = rng.normal(1.0, 0.40, n)
    Ra = Ra * noise

    # Clip to the actual observed Ra range from experiments (0.081 – 0.868 µm)
    Ra = np.clip(Ra, 0.081, 0.868)

    return pd.DataFrame({
        "Spindle_Speed": spindle_speed.round(1),
        "Feed_Rate":     feed_rate.round(1),
        "Depth_of_Cut":  depth_of_cut.round(3),
        "Ra":            Ra.round(4)
    })

df_ra = generate_ra_dataset()


# ─────────────────────────────────────────────
# 4. TRAIN AI MODELS ON SYNTHETIC RA DATASET
# ─────────────────────────────────────────────
@st.cache_resource
def train_ra_models(df, max_depth):
    """Trains LR and DT on the 1000-sample synthetic Ra dataset."""
    X = df[["Spindle_Speed", "Feed_Rate", "Depth_of_Cut"]]
    y = df["Ra"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)

    lr_preds = lr.predict(X_test)
    dt_preds = dt.predict(X_test)

    lr_metrics = {
        "R²":   round(r2_score(y_test, lr_preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, lr_preds)), 4),
        "MAE":  round(mean_absolute_error(y_test, lr_preds), 4),
    }
    dt_metrics = {
        "R²":   round(r2_score(y_test, dt_preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, dt_preds)), 4),
        "MAE":  round(mean_absolute_error(y_test, dt_preds), 4),
    }

    # Feature importance
    fi = dict(zip(["Spindle Speed", "Feed Rate", "Depth of Cut"],
                  dt.feature_importances_.round(4)))

    return lr, dt, lr_metrics, dt_metrics, fi, y_test.values, lr_preds, dt_preds


# ─────────────────────────────────────────────
# 5. SIDEBAR (controls — shown after files are loaded)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚙️ RDX20 Controls")

    st.markdown("**🧠 Algorithm**")
    selected_algorithm = st.selectbox(
        "Prediction Algorithm",
        ("Linear Regression", "Decision Tree"),
        label_visibility="collapsed"
    )

    st.markdown("**🌲 Decision Tree Depth**")
    max_depth = st.slider("Max Depth", 2, 12, 5)

    st.markdown("---")
    st.markdown("**🚨 Warning Thresholds**")
    spindle_thresh = st.slider("Spindle Load Warning (%)", 0.0, 120.0, 15.0)
    servo_thresh   = st.slider("Servo Load Warning (%)",   0.0, 120.0, 10.0)

    st.markdown("---")
    st.markdown("**🔬 Predict Surface Roughness**")
    pred_ss  = st.slider("Spindle Speed (rpm)",  400, 800, 600)
    pred_fr  = st.slider("Feed Rate (mm/min)",   100, 260, 180)
    pred_doc = st.slider("Depth of Cut (mm × 100)", 30, 120, 75)
    doc_val  = pred_doc / 100


# ─────────────────────────────────────────────
# 6. TRAIN & CACHE MODELS
# ─────────────────────────────────────────────
lr_ra, dt_ra, lr_met, dt_met, feat_imp, y_test, lr_preds_test, dt_preds_test = train_ra_models(df_ra, max_depth)

active_ra_model = lr_ra if selected_algorithm == "Linear Regression" else dt_ra
active_metrics  = lr_met if selected_algorithm == "Linear Regression" else dt_met


# ─────────────────────────────────────────────
# 8. HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:28px; margin-bottom:4px;'>
  ⚙️ RDX20 <span style='color:#00d4ff'>AI & Telemetry</span> Dashboard
</h1>
<p style='color:#64748b; font-size:13px; margin-bottom:20px;'>
  Aluminium CNC Machining · Surface Roughness Prediction · OEE Intelligence
</p>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 9. TOP KPI STRIP
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

pred_val = active_ra_model.predict([[pred_ss, pred_fr, doc_val]])[0]
pred_val = float(np.clip(pred_val, 0.081, 0.868))

ra_cls = "green" if pred_val < 0.25 else "orange" if pred_val < 0.50 else "red"

with k1:
    st.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>Active Algorithm</div>
      <div class='kpi-value' style='font-size:16px'>{selected_algorithm.replace(" ", "<br>")}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='kpi-card green'>
      <div class='kpi-label'>Model R²</div>
      <div class='kpi-value green'>{active_metrics['R²']}</div>
      <div class='kpi-sub'>on test split</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='kpi-card orange'>
      <div class='kpi-label'>RMSE</div>
      <div class='kpi-value orange'>{active_metrics['RMSE']}</div>
      <div class='kpi-sub'>µm Ra error</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class='kpi-card {ra_cls}'>
      <div class='kpi-label'>Predicted Ra</div>
      <div class='kpi-value {ra_cls}'>{pred_val:.3f}</div>
      <div class='kpi-sub'>µm surface roughness</div>
    </div>""", unsafe_allow_html=True)

with k5:
    total_alarms = len(df_alarms)
    st.markdown(f"""
    <div class='kpi-card red'>
      <div class='kpi-label'>Total Alarms</div>
      <div class='kpi-value red'>{total_alarms}</div>
      <div class='kpi-sub'>logged events</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 10. TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖  AI Surface Roughness Model",
    "🏭  Machining Run Analysis",
    "📈  Telemetry",
    "🔔  Alarm Logs"
])


# ══════════════════════════════════════════════
# TAB 1 — AI SURFACE ROUGHNESS MODEL
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-head'>// AI Model · Surface Roughness Prediction · Aluminium</div>",
                unsafe_allow_html=True)

    # ── Prediction result ──
    col_pred, col_quality = st.columns([1, 1])

    with col_pred:
        quality_label = ("🟢 Excellent" if pred_val < 0.20 else
                         "🟡 Good"      if pred_val < 0.30 else
                         "🟠 Moderate"  if pred_val < 0.50 else
                         "🔴 Poor")
        st.markdown(f"""
        <div class='pred-result'>
          <div class='p-label'>Predicted Surface Roughness Ra</div>
          <div class='p-val'>{pred_val:.4f}</div>
          <div class='p-unit'>µm · {quality_label}</div>
          <div style='margin-top:14px; font-size:12px; color:#64748b;'>
            Spindle: {pred_ss} rpm &nbsp;|&nbsp; Feed: {pred_fr} mm/min &nbsp;|&nbsp; DoC: {doc_val:.2f} mm
          </div>
        </div>""", unsafe_allow_html=True)

    with col_quality:
        # Ra quality gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_val,
            number={"suffix": " µm", "font": {"size": 28, "color": "#e2e8f0", "family": "Space Mono"}},
            gauge={
                "axis": {"range": [0, 0.9], "tickcolor": "#64748b",
                         "tickfont": {"color": "#64748b", "size": 10}},
                "bar":  {"color": "#00d4ff" if pred_val < 0.25 else "#ff6b35" if pred_val < 0.50 else "#ff4b4b",
                         "thickness": 0.25},
                "bgcolor": "#111827",
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 0.20], "color": "rgba(127,255,107,0.12)"},
                    {"range": [0.20, 0.30], "color": "rgba(0,212,255,0.08)"},
                    {"range": [0.30, 0.50], "color": "rgba(255,107,53,0.10)"},
                    {"range": [0.50, 0.90], "color": "rgba(255,75,75,0.15)"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 2}, "value": pred_val}
            },
            title={"text": "Ra Quality Gauge", "font": {"color": "#64748b", "size": 12}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
            font={"color": "#e2e8f0"}, height=220, margin=dict(t=30, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # ── Model Performance Metrics ──
    st.markdown("**Model Performance Comparison — Test Set (200 samples)**")
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    cols_met = [mc1, mc2, mc3, mc4, mc5, mc6]
    lr_color_map = {"R²": "#00d4ff", "RMSE": "#00d4ff", "MAE": "#00d4ff"}
    dt_color_map = {"R²": "#ff6b35", "RMSE": "#ff6b35", "MAE": "#ff6b35"}

    for i, (k, v) in enumerate(lr_met.items()):
        with cols_met[i]:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='m-label'>LR · {k}</div>
              <div class='m-val' style='color:#00d4ff'>{v}</div>
            </div>""", unsafe_allow_html=True)
    for i, (k, v) in enumerate(dt_met.items()):
        with cols_met[i + 3]:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='m-label'>DT · {k}</div>
              <div class='m-val' style='color:#ff6b35'>{v}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter + Feature Importance ──
    col_scatter, col_fi = st.columns([3, 2])

    with col_scatter:
        fig_scatter = go.Figure()
        mn, mx = min(y_test) - 0.1, max(y_test) + 0.1
        fig_scatter.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode='lines',
            line=dict(color='rgba(255,255,255,0.2)', dash='dash', width=1.5),
            name='Perfect Prediction', showlegend=True
        ))
        fig_scatter.add_trace(go.Scatter(
            x=y_test, y=lr_preds_test, mode='markers',
            marker=dict(color='rgba(0,212,255,0.5)', size=4),
            name='Linear Regression'
        ))
        fig_scatter.add_trace(go.Scatter(
            x=y_test, y=dt_preds_test, mode='markers',
            marker=dict(color='rgba(255,107,53,0.5)', size=4),
            name='Decision Tree'
        ))
        fig_scatter.update_layout(
            title="Actual vs Predicted Ra — Test Set",
            xaxis_title="Actual Ra (µm)", yaxis_title="Predicted Ra (µm)",
            paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0", family="DM Sans"),
            legend=dict(bgcolor="#111827", bordercolor="#1e3a5f"),
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
            height=320
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_fi:
        fi_df = pd.DataFrame({"Feature": list(feat_imp.keys()),
                              "Importance": list(feat_imp.values())}).sort_values("Importance")
        colors = ["#7fff6b", "#ff6b35", "#00d4ff"]
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation='h',
            marker=dict(color=colors[:len(fi_df)],
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            text=[f"{v*100:.1f}%" for v in fi_df["Importance"]],
            textposition='outside', textfont=dict(color="#e2e8f0", size=12)
        ))
        fig_fi.update_layout(
            title="Feature Importance (Decision Tree)",
            paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0", family="DM Sans"),
            xaxis=dict(gridcolor="#1e3a5f", range=[0, max(fi_df["Importance"]) * 1.3]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=320, margin=dict(r=60)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Ra Sensitivity Sweep ──
    st.markdown("---")
    st.markdown("**Ra Sensitivity — How each parameter affects surface roughness**")
    sweep_col1, sweep_col2, sweep_col3 = st.columns(3)

    def sweep_plot(param, values, fixed, title, color):
        preds = []
        for v in values:
            row = [fixed["ss"], fixed["fr"], fixed["doc"]]
            row[{"ss": 0, "fr": 1, "doc": 2}[param]] = v
            preds.append(float(np.clip(active_ra_model.predict([row])[0], 0.081, 0.868)))
        fig = go.Figure(go.Scatter(
            x=values, y=preds, mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color)
        ))
        fig.update_layout(
            title=title,
            paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0", family="DM Sans"),
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f", title="Ra (µm)"),
            height=220, margin=dict(t=40, b=30, l=50, r=20)
        )
        return fig

    fixed = {"ss": pred_ss, "fr": pred_fr, "doc": doc_val}

    with sweep_col1:
        st.plotly_chart(sweep_plot("ss", np.linspace(400, 800, 30), fixed,
                                   "Spindle Speed vs Ra", "#00d4ff"), use_container_width=True)
    with sweep_col2:
        st.plotly_chart(sweep_plot("fr", np.linspace(100, 260, 30), fixed,
                                   "Feed Rate vs Ra", "#ff6b35"), use_container_width=True)
    with sweep_col3:
        st.plotly_chart(sweep_plot("doc", np.linspace(0.3, 1.2, 30), fixed,
                                   "Depth of Cut vs Ra", "#7fff6b"), use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — MACHINING RUN ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-head'>// Machining Run Analysis · AI Quality Prediction</div>",
                unsafe_allow_html=True)

    # ── Run detection with glitch filter ──────────────────────
    # The cutting feed signal can produce spurious 1-2 row pulses
    # (≤2 s) at transitions that are not real machining runs.
    # We filter these out with a minimum duration threshold.
    SAMPLE_INTERVAL_S   = 0.5   # CSV is sampled every 0.5 s
    MIN_RUN_DURATION_S  = 5.0   # ignore runs shorter than this
    MIN_RUN_ROWS        = int(MIN_RUN_DURATION_S / SAMPLE_INTERVAL_S)  # = 10 rows

    df_telemetry['Block'] = (
        df_telemetry['Cutting_Signal'] != df_telemetry['Cutting_Signal'].shift(1)
    ).cumsum()
    cutting_phases = df_telemetry[df_telemetry['Cutting_Signal'] == 1]

    run_stats = cutting_phases.groupby('Block').agg(
        Start_Time=('Time', 'first'),
        End_Time=('Time', 'last'),
        Row_Count=('Time', 'count'),
        Feed_Rate=('Feed_Rate', 'mean'),
        Speed=('Speed', 'mean'),
        Servo_Load=('Servo_Load', 'mean'),
        Spindle_Load=('Spindle_Load', 'mean')
    ).reset_index(drop=True)

    # Drop glitch pulses (too short to be real cuts)
    run_stats = run_stats[run_stats['Row_Count'] >= MIN_RUN_ROWS].reset_index(drop=True)

    # Convert row count → actual seconds
    run_stats['Duration_Sec'] = (run_stats['Row_Count'] * SAMPLE_INTERVAL_S).round(1)
    run_stats = run_stats.drop(columns=['Row_Count'])

    run_stats.insert(0, 'Run_Number', run_stats.index + 1)

    # Predict Ra using the AI model (Spindle Speed, Feed Rate, approx Depth of Cut)
    # Depth of Cut approximated from Servo Load as a heuristic
    approx_doc = np.clip(run_stats['Servo_Load'] / 20.0 * 0.9 + 0.3, 0.3, 1.2)
    X_runs_ra = np.column_stack([run_stats['Speed'], run_stats['Feed_Rate'], approx_doc])
    run_stats['Predicted_Ra (µm)'] = np.clip(
        active_ra_model.predict(X_runs_ra), 0.081, 0.868
    ).round(3)

    st.write(f"Identified **{len(run_stats)} distinct cutting runs** · Algorithm: **{selected_algorithm}**")

    def color_ra(val):
        if isinstance(val, float):
            if val < 0.25: return 'color: #7fff6b; font-weight:700'
            if val < 0.50: return 'color: #ffd700; font-weight:700'
            return 'color: #ff4b4b; font-weight:700'
        return ''

    styled = (run_stats.style
              .applymap(color_ra, subset=['Predicted_Ra (µm)'])
              .format({
                  "Feed_Rate": "{:.1f}", "Speed": "{:.1f}",
                  "Servo_Load": "{:.1f}", "Spindle_Load": "{:.1f}",
                  "Duration_Sec": "{:.1f} s",
                  "Predicted_Ra (µm)": "{:.3f}"
              }))
    st.dataframe(styled, use_container_width=True)

    # Ra per run bar chart
    if len(run_stats) > 0:
        fig_runs = go.Figure()
        fig_runs.add_trace(go.Bar(
            x=run_stats['Run_Number'], y=run_stats['Predicted_Ra (µm)'],
            name=f'AI Model ({selected_algorithm})', marker_color='#00d4ff',
            text=run_stats['Predicted_Ra (µm)'].round(3),
            textposition='outside', textfont=dict(size=10)
        ))
        fig_runs.add_hline(y=0.50, line_dash="dash", line_color="#ff4b4b",
                           annotation_text="Ra = 0.50 µm Quality Limit")
        fig_runs.update_layout(
            title="Predicted Surface Roughness per Machining Run",
            xaxis_title="Run Number", yaxis_title="Predicted Ra (µm)",
            paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0", family="DM Sans"),
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
            legend=dict(bgcolor="#111827", bordercolor="#1e3a5f"),
            height=340
        )
        st.plotly_chart(fig_runs, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — TELEMETRY
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-head'>// Machine Telemetry · Real-Time Context</div>",
                unsafe_allow_html=True)

    # Warnings
    warnings = []
    if df_telemetry['Spindle_Load'].max() >= spindle_thresh:
        warnings.append(f"⚠️ Spindle load exceeded threshold! (Max: {df_telemetry['Spindle_Load'].max():.1f}%)")
    if df_telemetry['Servo_Load'].max() >= servo_thresh:
        warnings.append(f"⚠️ Servo load exceeded threshold! (Max: {df_telemetry['Servo_Load'].max():.1f}%)")

    if warnings:
        for w in warnings:
            st.error(w)
    else:
        st.success("✅ Machine operating within normal parameters.")

    col1, col2 = st.columns(2)

    def telem_fig(y_col, title, color, thresh=None, thresh_name=""):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_telemetry['Time'], y=df_telemetry[y_col],
            mode='lines', line=dict(color=color, width=1.5), name=y_col
        ))
        if thresh is not None:
            fig.add_hline(y=thresh, line_dash="dash", line_color="#ff4b4b",
                          annotation_text=thresh_name,
                          annotation_font_color="#ff4b4b")
        fig.update_layout(
            title=title,
            paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0", family="DM Sans"),
            xaxis=dict(gridcolor="#1e3a5f", tickangle=45),
            yaxis=dict(gridcolor="#1e3a5f"),
            height=280, margin=dict(t=40, b=60, l=50, r=20)
        )
        return fig

    with col1:
        st.plotly_chart(telem_fig("Speed", "Spindle Speed S (RPM)", "#7fff6b"), use_container_width=True)
        st.plotly_chart(telem_fig("Servo_Load", "Servo Load Current (%)", "#ff6b35",
                                  servo_thresh, "Warning Threshold"), use_container_width=True)
    with col2:
        st.plotly_chart(telem_fig("Feed_Rate", "Feed Rate F (mm/min)", "#00d4ff"), use_container_width=True)
        st.plotly_chart(telem_fig("Spindle_Load", "Spindle Motor Load (%)", "#a855f7",
                                  spindle_thresh, "Warning Threshold"), use_container_width=True)

    # Cutting signal overlay
    fig_cut = go.Figure()
    fig_cut.add_trace(go.Scatter(
        x=df_telemetry['Time'], y=df_telemetry['Cutting_Signal'],
        mode='lines', fill='tozeroy',
        line=dict(color='#00d4ff', width=1),
        fillcolor='rgba(0,212,255,0.12)', name='Cutting Active'
    ))
    fig_cut.update_layout(
        title="Cutting Signal (1 = Active Cut)",
        paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
        font=dict(color="#e2e8f0", family="DM Sans"),
        xaxis=dict(gridcolor="#1e3a5f", tickangle=45),
        yaxis=dict(gridcolor="#1e3a5f"),
        height=200, margin=dict(t=40, b=60, l=50, r=20)
    )
    st.plotly_chart(fig_cut, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — ALARM LOGS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-head'>// Machine Alarm Logs</div>",
                unsafe_allow_html=True)

    # Alarm summary KPIs
    ex_count  = len(df_alarms[df_alarms['AlarmKind'] == 'EX'])  if 'AlarmKind' in df_alarms.columns else 0
    opr_count = len(df_alarms[df_alarms['AlarmKind'] == 'OPR']) if 'AlarmKind' in df_alarms.columns else 0

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown(f"""<div class='kpi-card red'>
          <div class='kpi-label'>EX Alarms</div>
          <div class='kpi-value red'>{ex_count}</div>
        </div>""", unsafe_allow_html=True)
    with ac2:
        st.markdown(f"""<div class='kpi-card orange'>
          <div class='kpi-label'>OPR Alarms</div>
          <div class='kpi-value orange'>{opr_count}</div>
        </div>""", unsafe_allow_html=True)
    with ac3:
        st.markdown(f"""<div class='kpi-card'>
          <div class='kpi-label'>Total Alarms</div>
          <div class='kpi-value'>{len(df_alarms)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def color_alarms(val):
        if val == 'EX':  return 'background-color: rgba(255,75,75,0.2); color: #ff4b4b; font-weight:700'
        if val == 'OPR': return 'background-color: rgba(255,215,0,0.15); color: #ffd700; font-weight:700'
        return ''

    cols_to_show = [c for c in ['DateAndTimeOfOccurrence', 'AlarmKind', 'AlarmNumber',
                                 'AlarmMessage', 'TimeSpanOfOccurrence(minute)']
                    if c in df_alarms.columns]
    st.dataframe(
        df_alarms[cols_to_show].style.applymap(
            color_alarms, subset=['AlarmKind'] if 'AlarmKind' in df_alarms.columns else []
        ),
        use_container_width=True
    )
