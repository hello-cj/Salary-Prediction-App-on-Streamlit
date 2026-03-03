import streamlit as st
import pickle
import numpy as np

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered"
)

# -----------------------
# LOAD MODEL
# -----------------------
model = pickle.load(open("random_forest_regressor_salary_predictor_v1.pkl", "rb"))
MODEL_R2 = 0.9185
MODEL_PERCENT = MODEL_R2 * 100

# -----------------------
# PREMIUM CSS
# -----------------------
st.markdown("""
<style>

/* =============================
   APP BACKGROUND
============================= */
.stApp {
    background: #f9fafb; /* very light gray for soft contrast */
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* =============================
   MAIN CARD
============================= */
section.main > div {
    background: #ffffff;
    padding: 50px 50px;
    border-radius: 24px;
    max-width: 780px;
    margin: auto;
    box-shadow: 0 12px 40px rgba(0,0,0,0.05); /* soft premium float */
    transition: all 0.3s ease;
}

/* =============================
   TITLE & SUBTITLE
============================= */
h1 {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 35px;
}

/* =============================
   INPUTS (UNIFORM)
============================= */
label {
    font-weight: 600 !important;
    color: #334155 !important;
}

.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    background-color: #f8f9fa !important;  /* unified background */
    padding: 0.6em 0.8em !important;
    font-size: 15px;
    transition: all 0.2s ease;
}

/* Inner select box fix */
.stSelectbox div[data-baseweb="select"] > div:first-child {
    background-color: #f8f9fa !important;
}

.stNumberInput input:focus,
.stSelectbox div[data-baseweb="select"]:focus-within {
    border: 1px solid #2563eb !important;
    box-shadow: none !important;
}

/* =============================
   PREDICT BUTTON
============================= */
div[data-testid="stFormSubmitButton"] {
    display: flex;
    justify-content: center;
    margin-top: 10px; /* smaller gap to Education dropdown */
}

div[data-testid="stFormSubmitButton"] > button {
    width: 100%;               /* match input width */
    max-width: 320px;          /* same as dropdown */
    height: 3em;               /* clean oblong */
    border-radius: 18px;       /* soft oblong */
    background-color: #2563eb !important; /* solid professional blue */
    color: #ffffff !important;
    font-size: 16px;
    font-weight: 600;
    border: none;
    box-shadow: none !important; /* NO shadow */
    transition: all 0.2s ease;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1d4ed8 !important; /* darker blue on hover */
    transform: translateY(-1px);
}

/* =============================
   METRIC CARDS
============================= */
.metric-box {
    background: #ffffff;
    padding: 25px 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 0 0 rgba(0,0,0,0); /* no shadow */
    transition: all 0.25s ease;
}

.metric-box::before {
    content: "";
    display: block;
    height: 3px;
    width: 50px;
    margin: 0 auto 12px auto;
    background: #2563eb;
    border-radius: 4px;
}

.metric-box:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 0 rgba(0,0,0,0); /* no shadow on hover */
}

.metric-title {
    font-size: 12px;
    letter-spacing: 1px;
    color: #6b7280;
    margin-bottom: 8px;
    text-transform: uppercase;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
}

/* =============================
   FORM COLUMN SPACING
============================= */
.stForm > .row-widget.stNumberInput,
.stForm > .row-widget.stSelectbox {
    margin-bottom: 10px; /* closer spacing */
}

.stForm > .row-widget.stColumns {
    gap: 20px; /* space between side-by-side columns */
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# HEADER
# -----------------------
st.markdown("<h1>Salary Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered salary estimation based on profile data</div>", unsafe_allow_html=True)

# -----------------------
# FORM
# -----------------------
with st.form("salary_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65)

    with col2:
        experience = st.number_input("Years of Experience", min_value=0, max_value=40)

    education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])

    submit = st.form_submit_button("Predict Salary")

edu_master = 1 if education == "Master" else 0
edu_phd = 1 if education == "PhD" else 0

# -----------------------
# PREDICTION
# -----------------------
if submit:
    input_data = np.array([[age, experience, edu_master, edu_phd]])
    prediction = model.predict(input_data)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">EXPECTED SALARY</div>
            <div class="metric-value">₱{prediction:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">MODEL ACCURACY (R²)</div>
            <div class="metric-value">{MODEL_PERCENT:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)