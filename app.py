import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cognitive Drift Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SUPER PREMIUM CSS (Enhanced with new animations)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.85)),
                url("https://images.unsplash.com/photo-1620121692029-d088224ddc74");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container { padding-top: 2.5rem; padding-left: 3.5rem; padding-right: 3.5rem; }

#MainMenu, footer, header { visibility: hidden; }

/* Premium Sidebar */
[data-testid="stSidebar"] {
    background: rgba(5, 5, 15, 0.92) !important;
    border-right: 1px solid rgba(100, 180, 255, 0.15);
    backdrop-filter: blur(32px);
}

.signal-container {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(140, 200, 255, 0.15);
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 20px;
    transition: all 0.4s ease;
}

.signal-container:hover {
    border-color: rgba(140, 220, 255, 0.4);
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}

/* Glass Cards */
.glass-card {
    background: rgba(15, 20, 35, 0.65);
    border: 1px solid rgba(140, 200, 255, 0.2);
    padding: 34px 30px;
    border-radius: 28px;
    backdrop-filter: blur(24px);
    transition: all 0.45s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 10px 40px rgba(0,0,0,0.25);
}

.glass-card:hover {
    transform: translateY(-12px);
    border-color: rgba(160, 220, 255, 0.35);
    box-shadow: 0 25px 50px rgba(80, 160, 255, 0.15);
}

.metric-value {
    font-size: 58px;
    font-weight: 700;
    background: linear-gradient(90deg, #ffffff, #80b8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* New Animations for Added Features */
.ekg-container { animation: pulse 2s infinite; }
@keyframes pulse { 0% { opacity: 0.8; } 50% { opacity: 1; } 100% { opacity: 0.8; } }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA & MODEL 
# =========================================================
import os
import numpy as np

# Original + Safe Fix
if os.path.exists("/Users/gauravsuthar/Pictures/Codes/Cognitive Drift Analyzer AI/data/drift_data.csv"):
    df = pd.read_csv("/Users/gauravsuthar/Pictures/Codes/Cognitive Drift Analyzer AI/data/drift_data.csv")
    st.success("✅ CSV Loaded Successfully!")
else:
    st.warning("⚠️ CSV file not found. Using Demo Data.")
    df = pd.DataFrame({
        'response_latency': np.random.randint(20, 85, 150),
        'task_decay': np.random.randint(20, 85, 150),
        'focus_fragmentation': np.random.randint(10, 75, 150),
        'late_night_activity': np.random.randint(20, 95, 150),
        'goal_failure': np.random.randint(0, 20, 150),
        'withdrawal': np.random.randint(20, 85, 150),
        'consistency': np.random.randint(20, 85, 150),
        'cognitive_drift': np.random.choice([0, 1], 150)
    })

X = df.drop("cognitive_drift", axis=1)
y = df["cognitive_drift"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("""
<div style="font-size:32px; font-weight:700; margin-bottom:8px; letter-spacing:-1px;">Neural Signal Engine</div>
<div style="font-size:14.5px; color:rgba(255,255,255,0.7);">Real-time cognitive drift monitoring</div>
""", unsafe_allow_html=True)

def signal_block(title, desc, minv, maxv, default, key):
    st.sidebar.markdown(f"""
    <div class="signal-container">
        <div style="font-weight:600; margin-bottom:8px;">{title}</div>
        <div style="font-size:13px; color:rgba(255,255,255,0.6); line-height:1.5;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)
   
    value = st.sidebar.slider("", minv, maxv, default, key=key)
   
    if value > 70:
        st.sidebar.markdown('<div style="background:rgba(255,60,60,0.15); color:#FF6B6B; padding:6px 16px; border-radius:999px; font-size:13px; display:inline-block; margin:8px 0;">CRITICAL</div>', unsafe_allow_html=True)
    elif value > 40:
        st.sidebar.markdown('<div style="background:rgba(255,160,60,0.15); color:#FFB84D; padding:6px 16px; border-radius:999px; font-size:13px; display:inline-block; margin:8px 0;">ELEVATED</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div style="background:rgba(100,200,140,0.15); color:#64C48C; padding:6px 16px; border-radius:999px; font-size:13px; display:inline-block; margin:8px 0;">STABLE</div>', unsafe_allow_html=True)
    return value

# Inputs (Unchanged)
response_latency = signal_block("Response Latency Variance", "Measures instability in cognitive responsiveness patterns.", 0, 100, 32, "latency")
task_decay = signal_block("Task Completion Decay", "Tracks decline in sustained execution consistency.", 0, 100, 31, "task")
focus_fragmentation = signal_block("Focus Fragmentation", "Tracks excessive attention switching and instability.", 0, 100, 19, "focus")
late_night_activity = signal_block("Late Night Cognitive Activity", "Measures irregular recovery and cognitive fatigue cycles.", 0, 100, 48, "night")
goal_failure = signal_block("Micro Goal Failure Streak", "Measures repeated breakdown in micro execution loops.", 0, 20, 8, "goal")
withdrawal = signal_block("Interaction Withdrawal", "Measures reduced engagement and behavioral distancing.", 0, 100, 33, "withdrawal")
consistency = signal_block("Consistency Deviation", "Measures deviation from stable baseline behavioral patterns.", 0, 100, 36, "consistency")

# =========================================================
# PREDICTION & CLASSIFICATION (Unchanged)
# =========================================================
user_data = [[response_latency, task_decay, focus_fragmentation, late_night_activity, goal_failure, withdrawal, consistency]]
drift_score = model.predict_proba(user_data)[0][1] * 100
stability_score = 100 - drift_score

if drift_score > 80:
    classification = "Critical Cognitive Instability"
    col = "#FF4D4D"
elif drift_score > 60:
    classification = "High Behavioral Drift"
    col = "#FF9F4D"
else:
    classification = "Operational Stability Maintained"
    col = "#4DFFB5"

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div style="font-size:15px; letter-spacing:2px; opacity:0.6; margin-bottom:10px;">NEURO-INTELLIGENCE PLATFORM</div>
<div style="font-size:82px; font-weight:700; line-height:1; letter-spacing:-4px; background:linear-gradient(90deg,#fff,#a0c0ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
Cognitive Drift<br>Analyzer
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="font-size:21px; max-width:880px; color:rgba(255,255,255,0.8); margin-bottom:30px;">
AI-powered behavioral intelligence system detecting silent cognitive degradation in real-time.
</div>
""", unsafe_allow_html=True)

# Creator
st.markdown("""
<div style="display:flex; align-items:center; gap:18px; margin:30px 0 40px 0;">
    <img src="https://avatars.githubusercontent.com/u/583231?v=4" style="width:68px; height:68px; border-radius:50%; border:2px solid #8ab4ff;">
    <div>
        <div style="color:#aaa; font-size:14px;">Designed & Developed By</div>
        <div style="font-size:27px; font-weight:700;">Gaurav Suthar</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# METRIC CARDS (Unchanged)
# =========================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="glass-card"><div style="font-size:15px; opacity:0.7;">Cognitive Drift Score</div><div class="metric-value">{drift_score:.2f}%</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="glass-card"><div style="font-size:15px; opacity:0.7;">Stability Index</div><div class="metric-value">{stability_score:.2f}%</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="glass-card"><div style="font-size:15px; opacity:0.7;">Behavioral Status</div><div style="font-size:28px; font-weight:700; color:{col}; margin-top:12px;">{classification}</div></div>', unsafe_allow_html=True)

# =========================================================
# NEURAL INTERCONNECT WEB (Unchanged)
# =========================================================
st.markdown('<div style="font-size:38px; font-weight:700; margin:50px 0 25px 0;">Neural Interconnect Web</div>', unsafe_allow_html=True)

labels = ["Latency", "Task Decay", "Focus", "Night", "Goals", "Withdrawal", "Consistency"]
values = [response_latency, task_decay, focus_fragmentation, late_night_activity, goal_failure, withdrawal, consistency]

theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
x = np.cos(theta) * values
y = np.sin(theta) * values

fig_network = go.Figure()

for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        fig_network.add_trace(go.Scatter(
            x=[x[i], x[j]], y=[y[i], y[j]],
            mode='lines',
            line=dict(color='rgba(140,200,255,0.25)', width=1.5),
            hoverinfo='none'
        ))

fig_network.add_trace(go.Scatter(
    x=x, y=y,
    mode='markers+text',
    marker=dict(size=values, color=values, colorscale='Blues', showscale=True, line=dict(color='white', width=2)),
    text=labels,
    textposition="top center",
    textfont=dict(color="white", size=13),
    hovertemplate="<b>%{text}</b><br>Intensity: %{marker.size:.1f}<extra></extra>"
))

fig_network.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=520,
    margin=dict(l=0,r=0,t=0,b=0),
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)
st.plotly_chart(fig_network, use_container_width=True)

# =========================================================
# NEW: BEHAVIORAL GENOME RADAR + HUMAN OPERATING SYSTEM SCORE
# =========================================================
st.markdown('<div style="font-size:38px; font-weight:700; margin:50px 0 25px 0;">Behavioral Genome Radar</div>', unsafe_allow_html=True)

# Advanced Radar with multiple layers
radar_fig = go.Figure()

radar_fig.add_trace(go.Scatterpolar(
    r=values,
    theta=labels,
    fill='toself',
    name='Current State',
    line_color='#7db7ff'
))

radar_fig.add_trace(go.Scatterpolar(
    r=[v*0.85 for v in values],
    theta=labels,
    fill='toself',
    name='Baseline Genome',
    line_color='rgba(100,255,180,0.6)',
    opacity=0.7
))

radar_fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,100])),
    height=520,
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(radar_fig, use_container_width=True)

# Human Operating System Score
hos_score = (100 - drift_score + stability_score) / 2
st.markdown(f"""
<div class="glass-card">
    <div style="font-size:15px; opacity:0.7;">HUMAN OPERATING SYSTEM SCORE</div>
    <div style="font-size:58px; font-weight:700; color:#a0f0ff;">{hos_score:.1f}/100</div>
    <div style="color:#64ff9a;">Mental Throughput • Neural Efficiency • Emotional Resilience</div>
</div>
""", unsafe_allow_html=True)

# AI Personality Mapping
st.markdown('<div style="font-size:38px; font-weight:700; margin:50px 0 25px 0;">AI Personality Mapping</div>', unsafe_allow_html=True)

personality = "Hyper-Focused Bursts Performer" if focus_fragmentation < 25 and task_decay < 40 else \
              "Burnout-Prone Chaotic Worker" if late_night_activity > 60 else \
              "Analytical Deep Thinker" if consistency > 50 else "Adaptive Hybrid Executor"

st.markdown(f"""
<div class="glass-card">
    <div style="font-size:22px; color:#ffd700;">🧬 {personality}</div>
    <div style="margin-top:15px; color:rgba(255,255,255,0.85);">Based on your current behavioral genome signature.</div>
</div>
""", unsafe_allow_html=True)

# Human System Load
load = (sum(values) / len(values)) / 1.2
st.markdown(f"""
<div class="glass-card">
    <div style="font-size:15px; opacity:0.7;">HUMAN SYSTEM LOAD</div>
    <div style="font-size:52px; color:#ff6b6b;">{load:.1f}%</div>
    <div style="color:#ff9a9a;">CPU • Memory • Thermal</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PERFORMANCE EKG (Animated Neural Heartbeat)
# =========================================================
st.markdown('<div style="font-size:38px; font-weight:700; margin:50px 0 25px 0;">Performance EKG</div>', unsafe_allow_html=True)

ekg_x = list(range(10))
ekg_y = [drift_score/20 + np.sin(i) * (drift_score/40) for i in range(10)]

ekg = go.Figure(go.Scatter(x=ekg_x, y=ekg_y, mode='lines', line=dict(color='#00ff9d', width=4)))
ekg.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', title="Neural Activity Waveform")
st.plotly_chart(ekg, use_container_width=True)

# Focus Field Visualizer
st.markdown('<div style="font-size:38px; font-weight:700; margin:50px 0 25px 0;">Focus Field Visualizer</div>', unsafe_allow_html=True)

focus_fig = go.Figure()
focus_fig.add_trace(go.Scatter(x=[0, focus_fragmentation/50], y=[0, 1 - focus_fragmentation/100],
                               mode='lines+markers', line=dict(color='#00ddff', width=6), marker=dict(size=20)))
focus_fig.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', title="Magnetic Focus Field")
st.plotly_chart(focus_fig, use_container_width=True)

# Neural Overheat Warning
if drift_score > 65:
    st.error("🔥 NEURAL OVERHEAT WARNING - Cognitive Thermal Throttling Detected")
else:
    st.success("🟢 Neural Temperature Nominal")

# =========================================================
# RECOMMENDATIONS (Unchanged + Enhanced)
# =========================================================
st.markdown('<div style="font-size:38px; font-weight:700; margin:50px 0 25px 0;">Adaptive Recommendation Engine</div>', unsafe_allow_html=True)

recs = []
if response_latency > 55: recs.append(("Response Instability", "Reduce context switching. Use single-tab deep work mode."))
if task_decay > 60: recs.append(("Execution Decay", "Break tasks into 25-min focused sprints with 5-min breaks."))
if focus_fragmentation > 50: recs.append(("Attention Fragmentation", "Try the 5-4-3-2-1 grounding technique before starting work."))
if late_night_activity > 65: recs.append(("Recovery Deficit", "Implement strict 10 PM digital curfew for next 7 days."))
if goal_failure > 10: recs.append(("Micro-Goal Collapse", "Reduce goal size by 60% for the next 3 days."))
if withdrawal > 55: recs.append(("Social Withdrawal", "Schedule one 15-min meaningful conversation daily."))
if drift_score > 70: recs.append(("High Risk Alert", "Consider professional cognitive assessment if pattern persists."))

for title, text in recs:
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom:15px;">
        <div style="font-size:20px; font-weight:700; margin-bottom:10px;">{title}</div>
        <div style="color:rgba(255,255,255,0.85); line-height:1.6;">{text}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.5); font-size:14px;">
Cognitive Drift Analyzer • Ultra Premium Edition • Built with ❤️ by Gaurav Suthar
</div>
""", unsafe_allow_html=True)