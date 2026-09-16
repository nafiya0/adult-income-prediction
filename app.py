import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f7fbff, #f5f1ff, #f1fbf7);
}

.block-container {
    max-width: 1200px;
    padding-top: 30px;
    padding-bottom: 40px;
}


/* HERO */

.hero {
    background: linear-gradient(110deg, #dff3ff, #eee9ff, #f4eaff);
    border-radius: 25px;
    padding: 35px;
    text-align: center;
    border: 1px solid #e0e5f5;
    box-shadow: 0 10px 30px rgba(80, 90, 150, 0.10);
    margin-bottom: 30px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    color: #17245b;
}

.hero-title span {
    color: #6235c7;
}

.hero-subtitle {
    font-size: 18px;
    color: #5d6680;
    margin-top: 8px;
}

.hero-tagline {
    font-size: 15px;
    color: #6481d8;
    font-style: italic;
    margin-top: 10px;
}


/* SECTION HEADERS */

.section-header {
    background: rgba(255,255,255,0.90);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 18px;
    border: 1px solid #e1e6f2;
    box-shadow: 0 6px 18px rgba(70,80,130,0.06);
}

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #243574;
}

.section-description {
    font-size: 14px;
    color: #737c92;
    margin-top: 4px;
}


/* INPUTS */

label {
    color: #29365f !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"] {
    border-radius: 10px;
}

div[data-baseweb="select"] > div {
    border-radius: 10px;
}


/* TIP */

.tip {
    background: #e9faf4;
    border: 1px solid #c8ecdf;
    border-radius: 14px;
    padding: 15px 18px;
    margin-top: 20px;
    color: #246a5b;
    font-size: 14px;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    height: 60px;
    border: none;
    border-radius: 16px;
    background: linear-gradient(90deg, #7654e8, #3f83ed);
    color: white;
    font-size: 21px;
    font-weight: 700;
    box-shadow: 0 8px 22px rgba(76,91,220,0.25);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6845db, #3274dc);
    color: white;
}


/* RESULT */

.result {
    background: rgba(255,255,255,0.95);
    border: 2px dashed #b9c9f5;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-top: 28px;
    box-shadow: 0 8px 22px rgba(70,80,130,0.07);
}

.result-title {
    font-size: 18px;
    color: #68738d;
}

.result-value {
    font-size: 42px;
    font-weight: 800;
    color: #5738c7;
    margin-top: 5px;
}

.result-message {
    font-size: 15px;
    color: #68738d;
    margin-top: 5px;
}

.confidence {
    font-size: 15px;
    color: #53617d;
    margin-top: 10px;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #7c859b;
    font-size: 14px;
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid #dfe4ef;
}


/* HIDE STREAMLIT FOOTER */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    preprocessor = joblib.load("preprocessor.pkl")
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("random_forest_model.pkl")

    return preprocessor, scaler, model


preprocessor, scaler, model = load_model()


# =========================================================
# GET CATEGORIES FROM TRAINED MODEL
# =========================================================

categorical_cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

cat_encoder = preprocessor.named_transformers_["cat"]

categories = dict(
    zip(
        categorical_cols,
        cat_encoder.categories_
    )
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">💰 Adult Income <span>Prediction</span></div>'
    '<div class="hero-subtitle">Predict whether a person\'s annual income is ≤50K or >50K</div>'
    '<div class="hero-tagline">Data-Driven Insights for a Better Tomorrow</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# TWO COLUMNS
# =========================================================

left, right = st.columns(2, gap="large")


# =========================================================
# LEFT SIDE
# =========================================================

with left:

    st.markdown(
        '<div class="section-header">'
        '<div class="section-title">👤 Personal Information</div>'
        '<div class="section-description">Basic demographic details</div>'
        '</div>',
        unsafe_allow_html=True
    )

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=90,
        value=30
    )

    workclass = st.selectbox(
        "Workclass",
        categories["workclass"]
    )

    education = st.selectbox(
        "Education",
        categories["education"]
    )

    education_num = st.number_input(
        "Education Number",
        min_value=1,
        max_value=16,
        value=10
    )

    marital_status = st.selectbox(
        "Marital Status",
        categories["marital-status"]
    )

    relationship = st.selectbox(
        "Relationship",
        categories["relationship"]
    )

    race = st.selectbox(
        "Race",
        categories["race"]
    )

    sex = st.selectbox(
        "Sex",
        categories["sex"]
    )


# =========================================================
# RIGHT SIDE
# =========================================================

with right:

    st.markdown(
        '<div class="section-header">'
        '<div class="section-title">📊 Work & Financial Information</div>'
        '<div class="section-description">Employment and financial details</div>'
        '</div>',
        unsafe_allow_html=True
    )

    occupation = st.selectbox(
        "Occupation",
        categories["occupation"]
    )

    fnlwgt = st.number_input(
        "Final Weight (fnlwgt)",
        min_value=0,
        value=100000
    )

    capital_gain = st.number_input(
        "Capital Gain",
        min_value=0,
        value=0
    )

    capital_loss = st.number_input(
        "Capital Loss",
        min_value=0,
        value=0
    )

    hours_per_week = st.number_input(
        "Hours per Week",
        min_value=1,
        max_value=99,
        value=40
    )

    native_country = st.selectbox(
        "Native Country",
        categories["native-country"]
    )

    st.markdown(
        '<div class="tip">'
        '💡 <b>Tip:</b> Fill in the details and click '
        '<b>Predict Income</b> to get your prediction.'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

predict = st.button("✨  Predict Income")


# =========================================================
# PREDICTION
# =========================================================

if predict:

    input_data = pd.DataFrame([
        {
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "education-num": education_num,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours_per_week,
            "native-country": native_country
        }
    ])

    # Same preprocessing used during training
    X_encoded = preprocessor.transform(input_data)

    # Same scaling used during training
    X_scaled = scaler.transform(X_encoded)

    # Prediction
    prediction = model.predict(X_scaled)[0]

    # Probability
    probability = model.predict_proba(X_scaled)[0]

    confidence = max(probability) * 100


    if prediction == 1:

        result = ">50K"

        message = (
            "The model predicts an annual income above $50K."
        )

    else:

        result = "≤50K"

        message = (
            "The model predicts an annual income of $50K or below."
        )


    # =====================================================
    # RESULT CARD
    # =====================================================

    st.markdown(
        '<div class="result">'
        '<div class="result-title">🎯 Prediction Result</div>'
        f'<div class="result-value">{result}</div>'
        f'<div class="result-message">{message}</div>'
        f'<div class="confidence">Model confidence: <b>{confidence:.1f}%</b></div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '❤️ Built with Streamlit &nbsp; | &nbsp; '
    'Adult Income Prediction &nbsp; | &nbsp; '
    'Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)