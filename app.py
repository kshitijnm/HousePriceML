import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EstateAI | Intelligent Property Valuation",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("model/house_price_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")


# ============================================================
# ML ANALYTICS DATA
# ============================================================

@st.cache_data
def load_ml_analysis():

    df = pd.read_csv("dataset/Housing.csv")

    X = df.drop("price", axis=1)
    y = df["price"]

    # Same preprocessing used during the original model training
    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=int
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    analysis_model = LinearRegression()

    analysis_model.fit(
        X_train,
        y_train
    )

    y_pred = analysis_model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    return (
        df,
        X,
        y_test,
        y_pred,
        mae,
        rmse,
        r2
    )


(
    analysis_df,
    analysis_X,
    analysis_y_test,
    analysis_y_pred,
    analysis_mae,
    analysis_rmse,
    analysis_r2
) = load_ml_analysis()


# ============================================================
# HTML RENDERING HELPER
#
# IMPORTANT:
# Streamlit's native st.html() is used instead of
# st.markdown(..., unsafe_allow_html=True).
#
# This keeps the exact HTML/CSS frontend but prevents
# Streamlit from displaying the HTML source as plain code.
# ============================================================

def render_html(content):

    cleaned_lines = []

    for line in content.splitlines():

        cleaned_lines.append(
            line.lstrip()
        )

    cleaned_html = "\n".join(
        cleaned_lines
    )

    st.html(
        cleaned_html
    )


# ============================================================
# PREMIUM LUXURY CSS
# ============================================================

st.html(
"""
<style>

/* =========================================================
   IMPORT FONT
   ========================================================= */

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap'
);


/* =========================================================
   COLOR SYSTEM

   Obsidian       #08090C
   Surface        #101216
   Surface 2      #15171D
   Gold           #C9A86A
   Gold Light     #E5C98B
   Ivory          #F4F0E8
   Slate          #9297A3
   Emerald        #5FC39A
   ========================================================= */


/* =========================================================
   GLOBAL
   ========================================================= */

html,
body,
[class*="css"] {

    font-family: "DM Sans", sans-serif;

}


.stApp {

    background:

        radial-gradient(
            ellipse at 15% 0%,
            rgba(201, 168, 106, 0.065),
            transparent 32%
        ),

        radial-gradient(
            ellipse at 90% 20%,
            rgba(94, 195, 154, 0.035),
            transparent 30%
        ),

        #08090C;

    color: #F4F0E8;

}


.block-container {

    max-width: 1320px;

    padding-top: 28px;

    padding-bottom: 80px;

}


/* =========================================================
   HIDE STREAMLIT DEFAULT ELEMENTS
   ========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {

    background:
        rgba(8, 9, 12, 0.88);

    backdrop-filter:
        blur(18px);

}


/* =========================================================
   TOP NAVIGATION
   ========================================================= */

.top-nav {

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding:
        13px 18px;

    margin-bottom: 30px;

    background:
        rgba(16, 18, 22, 0.78);

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        14px;

    backdrop-filter:
        blur(20px);

}


.brand {

    display: flex;

    align-items: center;

    gap: 11px;

}


.brand-mark {

    width: 33px;

    height: 33px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 9px;

    background:

        linear-gradient(
            135deg,
            #DCC18A,
            #A98248
        );

    color:
        #111;

    font-family:
        "Playfair Display",
        serif;

    font-size: 18px;

    font-weight: 700;

    box-shadow:
        0 4px 15px rgba(201,168,106,0.18);

}


.brand-name {

    color:
        #F4F0E8;

    font-size:
        14px;

    font-weight:
        700;

    letter-spacing:
        0.3px;

}


.brand-tag {

    color:
        #727782;

    font-size:
        10px;

    letter-spacing:
        1.5px;

    text-transform:
        uppercase;

}


.nav-status {

    display: flex;

    align-items: center;

    gap: 8px;

    color:
        #777C87;

    font-size:
        10px;

    letter-spacing:
        1.2px;

    text-transform:
        uppercase;

}


.status-dot {

    width:
        6px;

    height:
        6px;

    border-radius:
        50%;

    background:
        #5FC39A;

    box-shadow:
        0 0 10px rgba(95,195,154,0.65);

}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    position: relative;

    overflow: hidden;

    min-height:
        515px;

    padding:
        72px 72px 68px;

    border-radius:
        28px;

    margin-bottom:
        70px;

    background:

        linear-gradient(
            110deg,
            #101216 0%,
            #0D0F13 52%,
            #11120F 100%
        );

    border:
        1px solid rgba(255,255,255,0.075);

    box-shadow:

        0 35px 90px rgba(0,0,0,0.42),

        inset 0 1px 0
        rgba(255,255,255,0.035);

}


/* Fine luxury grid */

.hero::before {

    content: "";

    position: absolute;

    inset: 0;

    background-image:

        linear-gradient(
            rgba(201,168,106,0.025) 1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            rgba(201,168,106,0.025) 1px,
            transparent 1px
        );

    background-size:
        55px 55px;

    mask-image:
        linear-gradient(
            to bottom right,
            black,
            transparent 70%
        );

}


/* Gold atmospheric light */

.hero::after {

    content: "";

    position: absolute;

    width:
        480px;

    height:
        480px;

    right:
        -170px;

    top:
        -180px;

    border-radius:
        50%;

    background:
        rgba(201,168,106,0.075);

    filter:
        blur(5px);

}


.hero-content {

    position:
        relative;

    z-index:
        2;

    max-width:
        850px;

}


.eyebrow {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        9px;

    color:
        #C9A86A;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        2.5px;

    text-transform:
        uppercase;

}


.eyebrow-line {

    width:
        25px;

    height:
        1px;

    background:
        #C9A86A;

}


.hero-title {

    margin-top:
        22px;

    color:
        #F4F0E8;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        clamp(48px, 6vw, 78px);

    line-height:
        1.02;

    font-weight:
        600;

    letter-spacing:
        -3px;

}


.hero-title em {

    color:
        #C9A86A;

    font-style:
        normal;

}


.hero-description {

    max-width:
        670px;

    margin-top:
        24px;

    color:
        #898E99;

    font-size:
        15px;

    line-height:
        1.9;

}


.hero-bottom {

    display:
        flex;

    flex-wrap:
        wrap;

    align-items:
        center;

    gap:
        10px;

    margin-top:
        42px;

}


.hero-chip {

    padding:
        9px 14px;

    color:
        #AEB2BA;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.07);

    border-radius:
        999px;

    font-size:
        11px;

}


/* =========================================================
   SECTION HEADER
   ========================================================= */

.section-eyebrow {

    color:
        #C9A86A;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        2.3px;

    text-transform:
        uppercase;

    margin-bottom:
        10px;

}


.section-title {

    color:
        #F4F0E8;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        34px;

    font-weight:
        600;

    letter-spacing:
        -0.8px;

    margin-bottom:
        7px;

}


.section-description {

    color:
        #747984;

    font-size:
        13px;

    line-height:
        1.7;

    margin-bottom:
        30px;

}


/* =========================================================
   INPUT PANEL
   ========================================================= */

.panel {

    background:
        linear-gradient(
            145deg,
            rgba(18,20,25,0.96),
            rgba(12,14,18,0.96)
        );

    border:
        1px solid rgba(255,255,255,0.07);

    border-radius:
        21px;

    padding:
        28px;

    margin-bottom:
        22px;

    box-shadow:
        0 20px 50px rgba(0,0,0,0.22);

}


.panel-heading {

    display:
        flex;

    align-items:
        center;

    gap:
        13px;

    margin-bottom:
        25px;

}


.panel-icon {

    width:
        38px;

    height:
        38px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        10px;

    background:
        rgba(201,168,106,0.08);

    border:
        1px solid rgba(201,168,106,0.15);

    color:
        #D7B878;

    font-size:
        16px;

}


.panel-title {

    color:
        #E9E5DD;

    font-size:
        15px;

    font-weight:
        700;

}


.panel-subtitle {

    color:
        #676C76;

    font-size:
        11px;

    margin-top:
        3px;

}


/* =========================================================
   STREAMLIT INPUTS
   ========================================================= */

label {

    color:
        #888D97 !important;

    font-size:
        11px !important;

    font-weight:
        600 !important;

}


div[data-baseweb="input"] {

    background:
        #0B0D11 !important;

    border:
        1px solid #252932 !important;

    border-radius:
        10px !important;

}


div[data-baseweb="input"]:focus-within {

    border-color:
        rgba(201,168,106,0.55) !important;

    box-shadow:
        0 0 0 1px rgba(201,168,106,0.15) !important;

}


div[data-baseweb="input"] input {

    color:
        #E8E4DC !important;

}


div[data-baseweb="select"] > div {

    background:
        #0B0D11 !important;

    border:
        1px solid #252932 !important;

    border-radius:
        10px !important;

}


div[data-baseweb="select"] > div:focus-within {

    border-color:
        rgba(201,168,106,0.55) !important;

}


div[data-baseweb="select"] span {

    color:
        #E8E4DC !important;

}


/* =========================================================
   PREDICTION BUTTON
   ========================================================= */

.stButton > button {

    height:
        61px;

    border-radius:
        12px;

    border:
        1px solid rgba(226,197,137,0.35);

    background:

        linear-gradient(
            135deg,
            #C8A76A,
            #A98248
        );

    color:
        #11120F;

    font-size:
        13px;

    font-weight:
        800;

    letter-spacing:
        1.1px;

    box-shadow:

        0 12px 30px
        rgba(201,168,106,0.16);

    transition:
        all 0.25s ease;

}


.stButton > button:hover {

    transform:
        translateY(-2px);

    background:

        linear-gradient(
            135deg,
            #E1C687,
            #BC9659
        );

    border-color:
        rgba(255,232,176,0.65);

    box-shadow:

        0 16px 35px
        rgba(201,168,106,0.24);

}


/* =========================================================
   RESULT
   ========================================================= */

.result {

    position:
        relative;

    overflow:
        hidden;

    padding:
        48px;

    margin-top:
        30px;

    margin-bottom:
        25px;

    border-radius:
        23px;

    background:

        linear-gradient(
            135deg,
            #11130F 0%,
            #161711 55%,
            #101410 100%
        );

    border:
        1px solid rgba(201,168,106,0.20);

    box-shadow:

        0 25px 70px rgba(0,0,0,0.40),

        inset 0 1px 0
        rgba(255,255,255,0.035);

}


.result::after {

    content:
        "";

    position:
        absolute;

    width:
        330px;

    height:
        330px;

    right:
        -150px;

    top:
        -160px;

    border-radius:
        50%;

    background:
        rgba(201,168,106,0.08);

    filter:
        blur(4px);

}


.result-content {

    position:
        relative;

    z-index:
        2;

}


.result-eyebrow {

    color:
        #C9A86A;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        2.5px;

}


.result-price {

    margin-top:
        10px;

    color:
        #F4F0E8;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        clamp(42px, 5vw, 68px);

    font-weight:
        600;

    letter-spacing:
        -2px;

}


.result-lakh {

    margin-top:
        4px;

    color:
        #D3B477;

    font-size:
        13px;

}


.result-note {

    max-width:
        600px;

    margin-top:
        18px;

    color:
        #777C86;

    font-size:
        11px;

    line-height:
        1.7;

}


/* =========================================================
   SUMMARY CARDS
   ========================================================= */

.summary-card {

    padding:
        22px;

    background:
        #101216;

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        16px;

    min-height:
        125px;

}


.summary-icon {

    color:
        #C9A86A;

    font-size:
        17px;

    margin-bottom:
        14px;

}


.summary-label {

    color:
        #696E78;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        1.5px;

    text-transform:
        uppercase;

}


.summary-value {

    margin-top:
        6px;

    color:
        #EAE6DE;

    font-size:
        20px;

    font-weight:
        600;

}


/* =========================================================
   WORKFLOW CARDS
   ========================================================= */

.workflow-card {

    min-height:
        205px;

    padding:
        24px;

    background:
        #101216;

    border:
        1px solid rgba(255,255,255,0.06);

    border-radius:
        17px;

    transition:
        all 0.25s ease;

}


.workflow-card:hover {

    transform:
        translateY(-3px);

    border-color:
        rgba(201,168,106,0.20);

}


.workflow-number {

    color:
        #C9A86A;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        27px;

    font-weight:
        600;

}


.workflow-title {

    margin-top:
        20px;

    color:
        #E5E1D9;

    font-size:
        13px;

    font-weight:
        700;

}


.workflow-description {

    margin-top:
        10px;

    color:
        #696E78;

    font-size:
        11px;

    line-height:
        1.75;

}


/* =========================================================
   MODEL SECTION
   ========================================================= */

.model-card {

    padding:
        32px;

    background:
        linear-gradient(
            145deg,
            #111318,
            #0D0F13
        );

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        19px;

}


.model-name {

    color:
        #E8E3D9;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        25px;

    font-weight:
        600;

}


.model-line {

    width:
        45px;

    height:
        2px;

    margin-top:
        17px;

    background:
        #C9A86A;

}


.model-description {

    margin-top:
        20px;

    color:
        #777C86;

    font-size:
        12px;

    line-height:
        1.9;

}


.model-info {

    background:
        #101216;

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        19px;

    padding:
        12px 25px;

}


.info-row {

    display:
        flex;

    justify-content:
        space-between;

    gap:
        20px;

    padding:
        17px 0;

    border-bottom:
        1px solid rgba(255,255,255,0.045);

}


.info-row:last-child {

    border-bottom:
        none;

}


.info-label {

    color:
        #666B75;

    font-size:
        10px;

    text-transform:
        uppercase;

    letter-spacing:
        1.2px;

}


.info-value {

    color:
        #DCD7CF;

    font-size:
        11px;

    font-weight:
        600;

    text-align:
        right;

}


/* =========================================================
   PERFORMANCE CARDS
   ========================================================= */

.performance-card {

    padding:
        27px;

    background:
        linear-gradient(
            145deg,
            #111318,
            #0D0F13
        );

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        17px;

    min-height:
        160px;

}


.performance-icon {

    color:
        #C9A86A;

    font-size:
        18px;

}


.performance-label {

    margin-top:
        22px;

    color:
        #686D77;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        1.5px;

}


.performance-value {

    margin-top:
        7px;

    color:
        #ECE7DE;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        30px;

    font-weight:
        600;

}


/* =========================================================
   GRAPH / VISUALIZATION CARDS
   ========================================================= */

.graph-card {

    padding:
        24px;

    background:
        linear-gradient(
            145deg,
            #111318,
            #0D0F13
        );

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        18px;

    margin-top:
        18px;

    margin-bottom:
        8px;

}


.graph-title {

    color:
        #E9E5DD;

    font-size:
        14px;

    font-weight:
        700;

}


.graph-description {

    margin-top:
        6px;

    color:
        #6F747E;

    font-size:
        11px;

    line-height:
        1.7;

}


/* =========================================================
   STREAMLIT METRICS
   ========================================================= */

[data-testid="stMetric"] {

    background:
        #101216;

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        16px;

    padding:
        20px;

}


[data-testid="stMetricLabel"] {

    color:
        #696E78 !important;

}


[data-testid="stMetricValue"] {

    color:
        #EAE6DE !important;

}


/* =========================================================
   EXPANDERS
   ========================================================= */

[data-testid="stExpander"] {

    background:
        #101216;

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        14px;

}


[data-testid="stExpander"] summary {

    color:
        #C9A86A !important;

}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {

    border:
        1px solid rgba(255,255,255,0.06);

    border-radius:
        12px;

}


/* =========================================================
   ALERTS
   ========================================================= */

[data-testid="stAlert"] {

    background:
        #101216;

    border:
        1px solid rgba(201,168,106,0.18);

}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 800px) {

    .hero {

        padding:
            48px 30px;

        min-height:
            auto;

    }

    .hero-title {

        font-size:
            45px;

        letter-spacing:
            -2px;

    }

    .section-title {

        font-size:
            28px;

    }

    .result {

        padding:
            35px 25px;

    }

}

</style>
""")


# ============================================================
# TOP NAVIGATION
# ============================================================

render_html("""
<div class="top-nav">

<div class="brand">

<div class="brand-mark">
E
</div>

<div>

<div class="brand-name">
EstateAI
</div>

<div class="brand-tag">
Intelligent Property Valuation
</div>

</div>

</div>

<div class="nav-status">

<div class="status-dot"></div>

MODEL ONLINE

</div>

</div>
""")


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero">

<div class="hero-content">

<div class="eyebrow">

<div class="eyebrow-line"></div>

AI-POWERED PROPERTY VALUATION

</div>

<div class="hero-title">

Know the value.<br>

<span>Before you decide.</span>

</div>

<div class="hero-description">

EstateAI combines historical housing data with
Multiple Linear Regression to generate an intelligent
property valuation based on the characteristics of
your home.

</div>

<div class="hero-bottom">

<div class="hero-chip">
◇ Machine Learning
</div>

<div class="hero-chip">
◇ Data Driven
</div>

<div class="hero-chip">
◇ Instant Valuation
</div>

<div class="hero-chip">
◇ Private Local Model
</div>

</div>

</div>

</div>
""")


# ============================================================
# PROPERTY INPUT HEADER
# ============================================================

render_html("""
<div class="section-eyebrow">
PROPERTY DETAILS
</div>

<div class="section-title">
Tell us about the property
</div>

<div class="section-description">
Provide the characteristics below. The trained model will
use these variables to estimate the property's market value.
</div>
""")


# ============================================================
# SPECIFICATIONS PANEL
# ============================================================

render_html("""
<div class="panel">

<div class="panel-heading">

<div class="panel-icon">
⌂
</div>

<div>

<div class="panel-title">
Property Specifications
</div>

<div class="panel-subtitle">
Physical characteristics of the property
</div>

</div>

</div>

</div>
""")


a1, a2, a3 = st.columns(
    3,
    gap="large"
)


with a1:

    area = st.number_input(
        "House Area",
        min_value=1,
        value=5000,
        step=100
    )


with a2:

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        value=3,
        step=1
    )


with a3:

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0,
        value=2,
        step=1
    )


a4, a5, a6 = st.columns(
    3,
    gap="large"
)


with a4:

    stories = st.number_input(
        "Stories",
        min_value=1,
        value=2,
        step=1
    )


with a5:

    parking = st.number_input(
        "Parking",
        min_value=0,
        value=1,
        step=1
    )


with a6:

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ]
    )


# ============================================================
# FEATURES PANEL
# ============================================================

render_html("""
<div class="panel">

<div class="panel-heading">

<div class="panel-icon">
◇
</div>

<div>

<div class="panel-title">
Property Features
</div>

<div class="panel-subtitle">
Additional characteristics considered by the model
</div>

</div>

</div>

</div>
""")


f1, f2, f3 = st.columns(
    3,
    gap="large"
)


with f1:

    mainroad = st.selectbox(
        "Main Road",
        [
            "yes",
            "no"
        ]
    )


with f2:

    guestroom = st.selectbox(
        "Guest Room",
        [
            "yes",
            "no"
        ]
    )


with f3:

    basement = st.selectbox(
        "Basement",
        [
            "yes",
            "no"
        ]
    )


f4, f5, f6 = st.columns(
    3,
    gap="large"
)


with f4:

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        [
            "yes",
            "no"
        ]
    )


with f5:

    airconditioning = st.selectbox(
        "Air Conditioning",
        [
            "yes",
            "no"
        ]
    )


with f6:

    prefarea = st.selectbox(
        "Preferred Area",
        [
            "yes",
            "no"
        ]
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

predict = st.button(
    "ESTIMATE PROPERTY VALUE"
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    new_house = pd.DataFrame({

        "area": [
            area
        ],

        "bedrooms": [
            bedrooms
        ],

        "bathrooms": [
            bathrooms
        ],

        "stories": [
            stories
        ],

        "parking": [
            parking
        ],

        "mainroad": [
            mainroad
        ],

        "guestroom": [
            guestroom
        ],

        "basement": [
            basement
        ],

        "hotwaterheating": [
            hotwaterheating
        ],

        "airconditioning": [
            airconditioning
        ],

        "prefarea": [
            prefarea
        ],

        "furnishingstatus": [
            furnishingstatus
        ]

    })


    new_house = pd.get_dummies(
        new_house,
        drop_first=True,
        dtype=int
    )


    new_house = new_house.reindex(
        columns=feature_names,
        fill_value=0
    )


    predicted_price = model.predict(
        new_house
    )[0]


    # ========================================================
    # RESULT
    # ========================================================

    render_html(
        f"""
<div class="result">

<div class="result-content">

<div class="result-eyebrow">
ESTIMATED PROPERTY VALUE
</div>

<div class="result-price">
₹ {predicted_price:,.2f}
</div>

<div class="result-lakh">
Approximately ₹ {predicted_price / 100000:.2f} Lakh
</div>

<div class="result-note">
Calculated using the trained Multiple Linear Regression
model and the property characteristics provided above.
</div>

</div>

</div>
"""
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    render_html("""
<div class="section-eyebrow">
PROPERTY SNAPSHOT
</div>
""")


    s1, s2, s3, s4 = st.columns(
        4,
        gap="medium"
    )


    summary_items = [

        (
            "⌂",
            "Area",
            f"{area:,}"
        ),

        (
            "□",
            "Bedrooms",
            str(bedrooms)
        ),

        (
            "◇",
            "Bathrooms",
            str(bathrooms)
        ),

        (
            "▣",
            "Parking",
            str(parking)
        )

    ]


    for column, item in zip(
        [s1, s2, s3, s4],
        summary_items
    ):

        icon, label, value = item

        with column:

            render_html(
                f"""
<div class="summary-card">

<div class="summary-icon">
{icon}
</div>

<div class="summary-label">
{label}
</div>

<div class="summary-value">
{value}
</div>

</div>
"""
            )


    st.write("")


    # ========================================================
    # COMPLETE DETAILS
    # ========================================================

    with st.expander(
        "◇   View complete property details"
    ):

        details = pd.DataFrame({

            "Property Feature": [

                "Area",
                "Bedrooms",
                "Bathrooms",
                "Stories",
                "Parking",
                "Main Road",
                "Guest Room",
                "Basement",
                "Hot Water Heating",
                "Air Conditioning",
                "Preferred Area",
                "Furnishing Status"

            ],

            "Value": [

                area,
                bedrooms,
                bathrooms,
                stories,
                parking,
                mainroad,
                guestroom,
                basement,
                hotwaterheating,
                airconditioning,
                prefarea,
                furnishingstatus

            ]

        })


        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.write("")
st.write("")


render_html("""
<div class="section-eyebrow">
THE PROCESS
</div>

<div class="section-title">
How EstateAI works
</div>

<div class="section-description">
A complete machine learning pipeline transforms property
characteristics into an estimated valuation.
</div>
""")


w1, w2, w3, w4, w5 = st.columns(
    5,
    gap="medium"
)


workflow_data = [

    (
        "01",
        "Property Input",
        "The user enters the physical and categorical characteristics of the property."
    ),

    (
        "02",
        "Preprocessing",
        "Categorical values are transformed into numerical features."
    ),

    (
        "03",
        "Regression Model",
        "The trained model processes the prepared feature vector."
    ),

    (
        "04",
        "Price Prediction",
        "The regression model calculates the estimated property value."
    ),

    (
        "05",
        "Result",
        "The predicted value is presented through the EstateAI interface."
    )

]


for column, workflow in zip(
    [w1, w2, w3, w4, w5],
    workflow_data
):

    number, title, description = workflow

    with column:

        render_html(
            f"""
<div class="workflow-card">

<div class="workflow-number">
{number}
</div>

<div class="workflow-title">
{title}
</div>

<div class="workflow-description">
{description}
</div>

</div>
"""
        )


# ============================================================
# MODEL
# ============================================================

st.write("")
st.write("")


render_html("""
<div class="section-eyebrow">
THE MACHINE LEARNING MODEL
</div>

<div class="section-title">
Multiple Linear Regression
</div>

<div class="section-description">
The algorithm powering EstateAI's house price prediction.
</div>
""")


model_left, model_right = st.columns(
    [1.45, 1],
    gap="large"
)


with model_left:

    render_html("""
<div class="model-card">

<div class="model-name">
Multiple Linear Regression
</div>

<div class="model-line"></div>

<div class="model-description">

Multiple Linear Regression is a supervised
machine learning algorithm used to predict
a continuous numerical value using multiple
independent variables.

<br><br>

For EstateAI, the target variable is
<b>house price</b>.

<br><br>

During training, the algorithm learns
coefficients from historical property data.
These learned coefficients are then used
to estimate the price of a new property.

</div>

</div>
""")


with model_right:

    render_html("""
<div class="model-info">

<div class="info-row">

<div class="info-label">
Model Type
</div>

<div class="info-value">
Multiple Linear Regression
</div>

</div>

<div class="info-row">

<div class="info-label">
Learning Type
</div>

<div class="info-value">
Supervised Learning
</div>

</div>

<div class="info-row">

<div class="info-label">
Problem Type
</div>

<div class="info-value">
Regression
</div>

</div>

<div class="info-row">

<div class="info-label">
Target Variable
</div>

<div class="info-value">
House Price
</div>

</div>

</div>
""")


# ============================================================
# MATHEMATICAL FORMULA
# ============================================================

st.write("")

with st.expander(
    "◇   View the mathematical foundation"
):

    st.write(
        "The Multiple Linear Regression model can be represented as:"
    )

    st.latex(
        r"""
        \hat{y}
        =
        \beta_0
        +
        \beta_1X_1
        +
        \beta_2X_2
        +
        \cdots
        +
        \beta_nX_n
        """
    )

    st.write(
        "Here, the model learns the coefficients β from "
        "the training dataset and uses them to estimate "
        "the house price."
    )


# ============================================================
# PERFORMANCE
# ============================================================

st.write("")
st.write("")


render_html("""
<div class="section-eyebrow">
MODEL PERFORMANCE
</div>

<div class="section-title">
Measured on unseen data
</div>

<div class="section-description">
The following metrics were calculated using the held-out
test dataset of 109 properties.
</div>
""")


p1, p2, p3 = st.columns(
    3,
    gap="medium"
)


performance_data = [

    (
        p1,
        "◎",
        "R² SCORE",
        f"{analysis_r2 * 100:.2f}%"
    ),

    (
        p2,
        "◇",
        "MEAN ABSOLUTE ERROR",
        f"₹{analysis_mae / 100000:.2f} L"
    ),

    (
        p3,
        "△",
        "ROOT MEAN SQUARE ERROR",
        f"₹{analysis_rmse / 100000:.2f} L"
    )

]


for column, icon, label, value in performance_data:

    with column:

        render_html(
            f"""
<div class="performance-card">

<div class="performance-icon">
{icon}
</div>

<div class="performance-label">
{label}
</div>

<div class="performance-value">
{value}
</div>

</div>
"""
        )


# ============================================================
# ML VISUAL ANALYTICS
# ============================================================

st.write("")
st.write("")


render_html("""
<div class="section-eyebrow">
MODEL VISUALS
</div>

<div class="section-title">
See how the model behaves
</div>

<div class="section-description">
These visualizations use the same held-out test data and
Multiple Linear Regression pipeline used for model evaluation.
</div>
""")


# ============================================================
# GRAPH 1 — ACTUAL VS PREDICTED
# ============================================================

render_html("""
<div class="panel">

<div class="panel-heading">

<div class="panel-icon">
↗
</div>

<div>

<div class="panel-title">
Actual vs Predicted Prices
</div>

<div class="panel-subtitle">
Points closer to the diagonal line represent more accurate predictions.
</div>

</div>

</div>

</div>
""")


fig1, ax1 = plt.subplots(
    figsize=(10, 5.5)
)

fig1.patch.set_facecolor(
    "#101216"
)

ax1.set_facecolor(
    "#101216"
)


ax1.scatter(
    analysis_y_test,
    analysis_y_pred,
    s=48,
    alpha=0.70,
    color="#C9A86A",
    edgecolors="none"
)


min_value = min(
    analysis_y_test.min(),
    analysis_y_pred.min()
)

max_value = max(
    analysis_y_test.max(),
    analysis_y_pred.max()
)


ax1.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="#F4F0E8",
    linestyle="--",
    linewidth=1.5,
    label="Perfect Prediction"
)


ax1.set_xlabel(
    "Actual Price",
    color="#A8ADB8",
    fontsize=11
)

ax1.set_ylabel(
    "Predicted Price",
    color="#A8ADB8",
    fontsize=11
)


ax1.tick_params(
    colors="#A8ADB8"
)


ax1.grid(
    alpha=0.12
)


for spine in ax1.spines.values():

    spine.set_color(
        "#252830"
    )


legend = ax1.legend(
    facecolor="#101216",
    edgecolor="#252830"
)


for text in legend.get_texts():

    text.set_color(
        "#F4F0E8"
    )


st.pyplot(
    fig1,
    use_container_width=True
)

plt.close(
    fig1
)


# ============================================================
# GRAPH 2 — RESIDUAL ANALYSIS
# ============================================================

render_html("""
<div class="panel">

<div class="panel-heading">

<div class="panel-icon">
△
</div>

<div>

<div class="panel-title">
Residual Analysis
</div>

<div class="panel-subtitle">
Residual = Actual Price − Predicted Price.
</div>

</div>

</div>

</div>
""")


residuals = (
    analysis_y_test -
    analysis_y_pred
)


fig2, ax2 = plt.subplots(
    figsize=(10, 5.5)
)

fig2.patch.set_facecolor(
    "#101216"
)

ax2.set_facecolor(
    "#101216"
)


ax2.scatter(
    analysis_y_pred,
    residuals,
    s=48,
    alpha=0.70,
    color="#C9A86A",
    edgecolors="none"
)


ax2.axhline(
    y=0,
    color="#F4F0E8",
    linestyle="--",
    linewidth=1.5
)


ax2.set_xlabel(
    "Predicted Price",
    color="#A8ADB8",
    fontsize=11
)

ax2.set_ylabel(
    "Residual (Actual − Predicted)",
    color="#A8ADB8",
    fontsize=11
)


ax2.tick_params(
    colors="#A8ADB8"
)


ax2.grid(
    alpha=0.12
)


for spine in ax2.spines.values():

    spine.set_color(
        "#252830"
    )


st.pyplot(
    fig2,
    use_container_width=True
)

plt.close(
    fig2
)


# ============================================================
# GRAPH 3 — FEATURE RELATIONSHIP WITH PRICE
# ============================================================

render_html("""
<div class="panel">

<div class="panel-heading">

<div class="panel-icon">
◇
</div>

<div>

<div class="panel-title">
Feature Relationship with Price
</div>

<div class="panel-subtitle">
Correlation shows the strength of a linear relationship with price.
It does not by itself prove causation.
</div>

</div>

</div>

</div>
""")


numeric_columns = [

    "price",
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"

]


correlation = (

    analysis_df[
        numeric_columns
    ]

    .corr()["price"]

    .drop("price")

    .sort_values()

)


fig3, ax3 = plt.subplots(
    figsize=(10, 5.2)
)

fig3.patch.set_facecolor(
    "#101216"
)

ax3.set_facecolor(
    "#101216"
)


bars = ax3.barh(

    correlation.index,

    correlation.values,

    color="#C9A86A"

)


ax3.set_xlabel(
    "Correlation with Price",
    color="#A8ADB8",
    fontsize=11
)


ax3.tick_params(
    colors="#A8ADB8"
)


ax3.grid(
    axis="x",
    alpha=0.12
)


for spine in ax3.spines.values():

    spine.set_color(
        "#252830"
    )


for bar, value in zip(
    bars,
    correlation.values
):

    ax3.text(

        value,

        bar.get_y()
        +
        bar.get_height() / 2,

        f" {value:.2f}",

        va="center",

        color="#F4F0E8",

        fontsize=10

    )


st.pyplot(
    fig3,
    use_container_width=True
)

plt.close(
    fig3
)


# ============================================================
# GRAPH INTERPRETATION
# ============================================================

render_html("""
<div class="section-description">

<strong>
How to read these visuals:
</strong>

<br><br>

Actual vs Predicted shows prediction alignment.

<br>

Residual Analysis shows the distribution of errors.

<br>

Feature Relationship provides a simple numerical view
of relationships between selected numerical features
and price.

</div>
""")


# ============================================================
# DATASET
# ============================================================

st.write("")
st.write("")


render_html("""
<div class="section-eyebrow">
DATASET
</div>

<div class="section-title">
The data behind EstateAI
</div>

<div class="section-description">
The model was trained using a housing dataset containing
545 property records and 13 predictor columns after encoding.
</div>
""")


d1, d2, d3 = st.columns(
    3,
    gap="medium"
)


with d1:

    st.metric(
        "Total Properties",
        "545"
    )


with d2:

    st.metric(
        "Training Properties",
        "436"
    )


with d3:

    st.metric(
        "Testing Properties",
        "109"
    )


# ============================================================
# ABOUT
# ============================================================

st.write("")


with st.expander(
    "◇   About this project"
):

    st.markdown(
"""
### EstateAI — Intelligent House Price Prediction

EstateAI is an AIML semester project demonstrating
a complete supervised machine learning workflow.

**Algorithm**

Multiple Linear Regression

**Learning Type**

Supervised Learning

**Problem Type**

Regression

**Target**

House Price

**Dataset**

Housing dataset containing 545 property records.

**Technology**

Python • Pandas • NumPy • Scikit-learn •
Matplotlib • Seaborn • Streamlit

**Machine Learning Pipeline**

Data Collection → Exploratory Data Analysis →
Preprocessing → Train/Test Split →
Model Training → Evaluation → Prediction →
Interactive Web Application
"""
    )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">

<div class="footer-brand">
ESTATEAI
</div>

Intelligent Property Valuation

<br>

Multiple Linear Regression • AIML Semester Project

<br><br>

Built with Python • Pandas • Scikit-learn • Streamlit

</div>
""")