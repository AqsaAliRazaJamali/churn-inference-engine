import streamlit as st
import os

# Page configuration
st.set_page_config(page_title="Customer Churn Intelligence Dashboard", page_icon="📊", layout="wide")

# --- ADVANCED UI EMULATION & THEMING LAYER ---
st.markdown("""
<style>

/*==================================================
                GEMINI BACKGROUND
==================================================*/

.stApp{
    background:
        radial-gradient(circle at 10% 10%, rgba(0,170,255,.15), transparent 35%),
        radial-gradient(circle at 90% 0%, rgba(80,120,255,.10), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(0,170,255,.08), transparent 45%),
        linear-gradient(180deg,#101826 0%,#0d1117 60%,#090c10 100%);
    background-attachment:fixed;
}

/*==================================================
        HEADER FIX (BLENDS UPPER PORTION)
==================================================*/

div[data-testid="stHeader"] {
    background: transparent !important;
}

/*==================================================
            SIDEBAR CUSTOM DESIGN OVERHAUL
==================================================*/

div[data-testid="stSidebar"], 
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #101826 0%, #0d1117 100%) !important;
    border-right: 1px solid rgba(0, 242, 254, 0.15) !important;
    box-shadow: 4px 0 25px rgba(0, 0, 0, 0.5);
}

/* Sidebar navigation links and headers */
div[data-testid="stSidebar"] h1, 
div[data-testid="stSidebar"] h2, 
div[data-testid="stSidebar"] h3, 
div[data-testid="stSidebar"] p, 
div[data-testid="stSidebar"] span, 
div[data-testid="stSidebar"] label {
    color: #F8FAFC !important;
}

/* Active navigation state coloring */
div[data-testid="stSidebarNav"] ul li div a span {
    color: #E2E8F0 !important;
    font-weight: 500;
}

div[data-testid="stSidebarNav"] ul li[data-selected="true"] {
    background: rgba(0, 242, 254, 0.08) !important;
    border-left: 3px solid #00F2FE !important;
}

div[data-testid="stSidebarNav"] ul li[data-selected="true"] span {
    color: #00F2FE !important;
    font-weight: 700 !important;
}

/*==================================================
                    TEXT
==================================================*/

h1,h2,h3,h4,h5,h6,
p,span,label{
    color:#F8FAFC !important;
}

/*==================================================
                METRIC STYLING
==================================================*/

div[data-testid="stMetricValue"] {
    color: #00f2fe !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
    color: #8a99ad !important;
}

/*==================================================
            DIAGNOSTIC STATUS CARD (NEON CYAN)
==================================================*/

.status-card-success {
    background: rgba(20, 24, 34, 0.85) !important;
    border: 1px solid rgba(0, 242, 254, 0.65) !important;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 
        0 0 10px rgba(0, 242, 254, 0.40),
        0 0 25px rgba(0, 242, 254, 0.20),
        inset 0 0 15px rgba(0, 242, 254, 0.08);
    color: #F8FAFC !important;
    backdrop-filter: blur(18px);
}

</style>
""", unsafe_allow_html=True)

# Main Title Header
# Main Title Header
st.markdown("""
<div style="margin-bottom: 25px;">
    <h1 style="
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 800;
        font-size: 2.65rem;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #FFFFFF 30%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 242, 254, 0.25);
        display: flex;
        align-items: center;
        gap: 15px;
        margin: 0;
    ">
        <span style="
            -webkit-text-fill-color: initial; 
            text-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        ">📊</span> 
        Customer Churn Intelligence Dashboard
    </h1>
    <p style="
        color: #8a99ad; 
        font-size: 15px; 
        margin-top: 8px; 
        margin-bottom: 0;
        letter-spacing: 0.2px;
    ">
        Real-time tracking environment and operational binary classification monitoring control portal.
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='border-color: rgba(0, 242, 254, 0.15); margin-top: 0; margin-bottom: 30px;'>", unsafe_allow_html=True)
st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

# Top Metrics Row
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

with metrics_col1:
    st.metric(label="Model Status", value="Operational / Active")

with metrics_col2:
    st.metric(label="Target Pipeline Framework", value="Gradient Boosting Array")

with metrics_col3:
    st.metric(label="Inference Latency Target", value="< 15ms")

st.markdown("<br><br>", unsafe_allow_html=True)

# Model Environment Diagnostics Section
st.markdown('<h3 style="color:#ffffff; font-family:system-ui; font-weight:700;">Model Environment Diagnostics</h3>', unsafe_allow_html=True)

# Verify model assets safely to drive the status container dynamically
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
MODEL_PATH = 'models/churn_model.pkl'

if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
    st.markdown("""
    <div class="status-card-success">
        <span style="color: #00F2FE; font-weight: bold; margin-right: 8px; text-shadow: 0 0 8px rgba(0,242,254,0.6);">✓</span> 
        System core check passed: Production serialization pipelines are correctly compiled and located inside /models.
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Active production models not found. Please verify binaries inside your /models directory.")