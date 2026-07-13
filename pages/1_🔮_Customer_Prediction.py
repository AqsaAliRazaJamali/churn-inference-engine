import sys
import os

# Dynamically force the root directory into the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Inference Engine", page_icon="🔮", layout="wide")

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

/* Sidebar widgets support (nested selectboxes/inputs) */
div[data-testid="stSidebar"] .stSelectbox > div > div,
div[data-testid="stSidebar"] .stNumberInput input {
    background: #151b26 !important;
    border: 1px solid rgba(0, 242, 254, 0.2) !important;
}


/*==================================================
                    TEXT
==================================================*/

h1,h2,h3,h4,h5,h6,
p,span,label{
    color:#F8FAFC !important;
}


/* Labels above inputs */

.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stCheckbox label,
.stRadio label{

    color:#FFFFFF !important;

    font-size:15px !important;

    font-weight:700 !important;

    letter-spacing:.3px;
}


/*==================================================
            SECTION HEADER CARDS
==================================================*/

.section-card{

    background:rgba(20,24,34,.82);

    backdrop-filter:blur(18px);

    border-radius:14px;

    padding:20px;

    border:1px solid rgba(0,242,254,.65);

    box-shadow:
        0 0 10px rgba(0,242,254,.40),
        0 0 25px rgba(0,242,254,.20),
        inset 0 0 15px rgba(0,242,254,.08);

    margin-bottom:20px;
}

.section-card-fin{

    background:rgba(20,24,34,.82);

    backdrop-filter:blur(18px);

    border-radius:14px;

    padding:20px;

    border:1px solid rgba(255,0,127,.65);

    box-shadow:
        0 0 10px rgba(255,0,127,.40),
        0 0 25px rgba(255,0,127,.20),
        inset 0 0 15px rgba(255,0,127,.08);

    margin-bottom:20px;
}


/*==================================================
                    METRIC BOX
==================================================*/

.metric-box{

    background:rgba(24,30,42,.85);

    border:1px solid rgba(255,255,255,.08);

    border-radius:12px;

    backdrop-filter:blur(15px);

    box-shadow:
        0 8px 20px rgba(0,0,0,.35);

    padding:20px;

    text-align:center;
}


/*==================================================
                SELECT BOXES
==================================================*/

.stSelectbox > div > div{

    background:#151b26 !important;

    border:1px solid rgba(0,242,254,.25) !important;

    border-radius:10px !important;

    color:white !important;

    transition:.3s;

    box-shadow:
        0 0 8px rgba(0,242,254,.08);
}


/*==================================================
                NUMBER INPUTS
==================================================*/

.stNumberInput input{

    background:#151b26 !important;

    color:white !important;

    border:1px solid rgba(0,242,254,.25) !important;

    border-radius:10px !important;

    box-shadow:
        0 0 8px rgba(0,242,254,.08);

    transition:.3s;
}


/*==================================================
                INPUT FOCUS
==================================================*/

.stSelectbox > div > div:focus-within,
.stNumberInput input:focus{

    border:1px solid #00F2FE !important;

    box-shadow:
        0 0 8px rgba(0,242,254,.60),
        0 0 20px rgba(0,242,254,.30) !important;
}


/*==================================================
                    BUTTON
==================================================*/

div.stButton > button:first-child{

    background:linear-gradient(135deg,#ff007f,#7928ca);

    color:white;

    border:none;

    border-radius:10px;

    font-weight:700;

    width:100%;

    transition:.3s;

    box-shadow:
        0 0 18px rgba(255,0,127,.40);
}

div.stButton > button:first-child:hover{

    transform:translateY(-2px);

    box-shadow:
        0 0 28px rgba(255,0,127,.70);

    background:linear-gradient(135deg,#7928ca,#ff007f);
}


/*==================================================
            STREAMLIT CONTAINERS
==================================================*/

div[data-testid="stVerticalBlock"]{

    background:transparent;
}

div[data-testid="stHorizontalBlock"]{

    background:transparent;
}


/*==================================================
            DROPDOWN TEXT COLOR
==================================================*/

.stSelectbox *{

    color:white !important;
}


/*==================================================
            INPUT PLACEHOLDER
==================================================*/

input::placeholder{

    color:#B0BEC5 !important;
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
        ">🔮</span> 
        Live Customer Churn Risk Inference Engine
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
st.markdown('<p style="color:#8a99ad; font-size:16px; margin-bottom:30px;">Input customer contract metrics below to compute operational real-time churn probabilities through the Gradient Boosting array.</p>', unsafe_allow_html=True)

# Check for production binaries
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
MODEL_PATH = 'models/churn_model.pkl'

if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODEL_PATH):
    st.error("⚠️ Active production models not found. Please run the training pipeline first.")
    st.stop()

@st.cache_resource
def load_inference_artifacts():
    """Loads and caches models to keep inference under 15ms."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    return preprocessor, model, explainer

preprocessor, model, explainer = load_inference_artifacts()

# Layout entry fields into clear columns wrapped in styled headers
st.markdown("""
<div class="section-card-fin">
<h3 style="
margin-top:0;
color:#FF007F;
font-weight:700;
text-shadow:
0 0 8px rgba(255,0,127,.8),
0 0 20px rgba(255,0,127,.4);
">
📋 Core Customer & Service Profile
</h3>
</div>
""", unsafe_allow_html=True)
form_col1, form_col2, form_col3 = st.columns(3)

with form_col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen Status", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with form_col2:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])

with form_col3:
    online_security = st.selectbox("Online Security Addon", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup Addon", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support Addon", ["No", "Yes", "No internet service"])
    st.caption("Multimedia Services")
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], label_visibility="collapsed")
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-card-fin">
<h3 style="
margin-top:0;
color:#FF007F;
font-weight:700;
text-shadow:
0 0 8px rgba(255,0,127,.8),
0 0 20px rgba(255,0,127,.4);
">
💳 Account & Financial Configuration
</h3>
</div>
""", unsafe_allow_html=True)
fin_col1, fin_col2, fin_col3 = st.columns(3)

with fin_col1:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
with fin_col2:
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
    payment_method = st.selectbox("Payment Type", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
with fin_col3:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)

st.markdown("<br>", unsafe_allow_html=True)

# Submit button trigger
if st.button("Run Predictive Diagnostic Pipeline", type="primary"):
    input_data = pd.DataFrame([{
        'Gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])
    
    try:
        processed_features = preprocessor.transform(input_data)
        
        # Safe feature name fallback
        try:
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            else:
                raise AttributeError
        except AttributeError:
            if hasattr(preprocessor, 'steps'):
                try:
                    feature_names = preprocessor[:-1].get_feature_names_out()
                except Exception:
                    feature_names = [f"Feature {i}" for i in range(processed_features.shape[1])]
            else:
                feature_names = [f"Feature {i}" for i in range(processed_features.shape[1])]
        
        probabilities = model.predict_proba(processed_features)[0]
        churn_prob = probabilities[1]
        
        st.markdown('<h3 style="color:#ffffff; margin-top:30px; font-family:system-ui;">Core Inference Diagnostic Output</h3>', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(f"""
                <div class="metric-box">
                    <p style="color:#8a99ad; font-size:14px; margin:0; text-transform:uppercase; letter-spacing:1px;">Calculated Churn Probability</p>
                    <p style="color:#00f2fe; font-size:48px; font-weight:bold; margin:10px 0;">{churn_prob * 100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
        with res_col2:
            if churn_prob < 0.3:
                st.markdown("""
                    <div class="metric-box" style="border-color:#00ff87; background:rgba(0,255,135,0.05);">
                        <p style="color:#00ff87; font-size:18px; font-weight:bold; margin:0;">🟢 LOW CHURN RISK PROFILE</p>
                        <p style="color:#8a99ad; font-size:14px; margin-top:10px;">The client exhibits strong behavioral retention loops. Standard service parameters recommended.</p>
                    </div>
                """, unsafe_allow_html=True)
            elif churn_prob < 0.7:
                st.markdown("""
                    <div class="metric-box" style="border-color:#f9d423; background:rgba(249,212,35,0.05);">
                        <p style="color:#f9d423; font-size:18px; font-weight:bold; margin:0;">🟡 MEDIUM RISK DETECTED</p>
                        <p style="color:#8a99ad; font-size:14px; margin-top:10px;">Account parameters destabilizing. Proactive loyalty offers advised.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="metric-box" style="border-color:#ff0055; background:rgba(255,0,85,0.05); box-shadow: 0 0 20px rgba(255,0,85,0.15);">
                        <p style="color:#ff0055; font-size:18px; font-weight:bold; margin:0;">🔴 HIGH CHURN HAZARD ALERT</p>
                        <p style="color:#8a99ad; font-size:14px; margin-top:10px;">Critical system exit vectors triggered. Immediate customer success override required.</p>
                    </div>
                """, unsafe_allow_html=True)
                
        # --- EXPLAINABLE AI VISUALIZATION ---
        st.markdown('<h3 style="color:#ffffff; margin-top:40px; font-family:system-ui;">Explanatory AI Interface</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color:#8a99ad; font-size:14px;">The waterfall array charts local feature impact weights affecting the active inference sequence.</p>', unsafe_allow_html=True)
        
        shap_values = explainer(processed_features)
        
        # Darken and style the Matplotlib container
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0d0f12')
        ax.set_facecolor('#141822')
        
        if len(shap_values.shape) == 3:
            shap.plots.waterfall(shap.Explanation(
                values=shap_values.values[0][:, 1],
                base_values=shap_values.base_values[0][1],
                data=processed_features[0],
                feature_names=feature_names
            ), max_display=8, show=False)
        else:
            shap.plots.waterfall(shap.Explanation(
                values=shap_values.values[0],
                base_values=shap_values.base_values[0],
                data=processed_features[0],
                feature_names=feature_names
            ), max_display=8, show=False)
            
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Execution dynamic matrix transformation error: {e}")