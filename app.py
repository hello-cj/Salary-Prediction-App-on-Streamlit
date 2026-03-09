import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
# LOAD DATASET
# -----------------------
df = pd.read_csv("Salary Data.csv")

# -----------------------
# PREMIUM CSS
# -----------------------
st.markdown("""
<style>

/* APP BACKGROUND */
.stApp {
    background: #f9fafb;
}

/* MAIN CARD */
section.main > div {
    background: #ffffff;
    padding: 50px 50px;
    border-radius: 24px;
    max-width: 780px;
    margin: auto;
    box-shadow: 0 12px 40px rgba(0,0,0,0.05);
}

/* TITLE */
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

/* INPUTS */
label {
    font-weight: 600 !important;
    color: #334155 !important;
}

.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    background-color: #f8f9fa !important;
}

/* BUTTON */
div[data-testid="stFormSubmitButton"] {
    display: flex;
    justify-content: center;
}

div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    max-width: 320px;
    height: 3em;
    border-radius: 18px;
    background-color: #2563eb !important;
    color: white !important;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

/* METRIC CARDS */
.metric-box {
    background: #ffffff;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
    border: 1px solid #e5e7eb;
}

.metric-title {
    font-size: 12px;
    letter-spacing: 1px;
    color: #6b7280;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
    padding-top: 40px;
}

div[role="radiogroup"] > label {
    padding: 10px 12px;
    border-radius: 10px;
    margin-bottom: 4px;
}

div[role="radiogroup"] > label:hover {
    background: #f1f5f9;
}
            
/* SIDEBAR STYLE */

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
    padding-top: 40px;
}

/* remove red radio highlight */

div[role="radiogroup"] label {
    background: transparent !important;
}

/* selectbox cleaner */

.stSelectbox div[data-baseweb="select"] {
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# SIDEBAR NAVIGATION
# -----------------------
st.sidebar.markdown("##  Salary AI")

page = st.sidebar.radio(
    "Navigation",
    ["Dataset", "Correlations","Prediction"]
)

# -----------------------
# PREDICTION PAGE
# -----------------------
if page == "Prediction":

    st.markdown("<h1>Salary Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-powered salary estimation based on profile data</div>", unsafe_allow_html=True)

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

    if submit:

        input_data = np.array([[age, experience, edu_master, edu_phd]])

        prediction = model.predict(input_data)[0]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">EXPECTED SALARY</div>
                <div class="metric-value">${prediction:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">MODEL ACCURACY</div>
                <div class="metric-value">{MODEL_PERCENT:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------
# DATASET PAGE
# -----------------------
elif page == "Dataset":

    st.title("Salary Dataset")

    st.write("This dataset was used to train the salary prediction model.")

    st.dataframe(df, use_container_width=True)

    st.subheader("Dataset Info")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

# -----------------------
# CORRELATIONS PAGE
# -----------------------
elif page == "Correlations":

    st.title("Data Correlations & Relationships")

    st.write("Visualization of relationships between dataset variables.")

    # HEATMAP
    st.subheader("Correlation Matrix")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    col1, col2 = st.columns(2)

    # AGE VS SALARY
    with col1:
        st.subheader("Age vs Salary")

        fig2, ax2 = plt.subplots(figsize=(5,3))

        sns.scatterplot(
            data=df,
            x="Age",
            y="Salary",
            ax=ax2
        )

        st.pyplot(fig2)

    # EXPERIENCE VS SALARY
    with col2:
        st.subheader("Experience vs Salary")

        fig3, ax3 = plt.subplots(figsize=(5,3))

        sns.scatterplot(
            data=df,
            x="Years of Experience",
            y="Salary",
            ax=ax3
        )

        st.pyplot(fig3)

    st.divider()

    st.subheader("Salary vs Education Level")

    fig4, ax4 = plt.subplots(figsize=(5,3))

    sns.boxplot(
        data=df,
        x="Education Level",
        y="Salary",
        ax=ax4
    )

    st.pyplot(fig4)

    st.markdown("""
        ### Key Insights

        • Salary shows a strong positive relationship with **Age**.

        • Salary increases significantly with **Years of Experience**.

        • Employees with **higher education levels tend to earn higher salaries**.
    """)