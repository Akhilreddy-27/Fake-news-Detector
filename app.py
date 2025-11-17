import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load trained pipeline
model = joblib.load("fake_news_logreg.pkl")

# --- Page configuration ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# --- Custom CSS for styling ---
st.markdown(
    """
<style>
/* Background with mixed bright + dark gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1f25 25%, #23303f 50%, #1b1f27 75%, #0f0f12 100%);
    background-attachment: fixed;
    color: #ffffff;

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e1e, #111111);
    color: #ffffff;
    box-shadow: 0 0 15px rgba(255,255,255,0.15);
}

/* Input fields / cards */
.stTextInput, .stTextArea, .stSelectbox, .stDateInput {
    background: rgba(255, 255, 255, 0.07) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    padding: 12px !important;
    color: #ffffff !important;
    backdrop-filter: blur(6px);
    transition: 0.3s ease;
}

/* Hover effects */
.stTextInput:hover, .stTextArea:hover, .stSelectbox:hover, .stDateInput:hover {
    background: rgba(255,255,255,0.15) !important;
    border-color: #00eaff !important;
    box-shadow: 0 0 12px rgba(0,234,255,0.5);
}

/* Labels */
div.stTextInput > label, div.stTextArea > label, div.stSelectbox > label, div.stDateInput > label {
    font-weight: 600;
    color: #00eaff;
    text-shadow: 0 0 6px rgba(0,234,255,0.6);
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s ease;
    border: none;
    box-shadow: 0 4px 12px rgba(255,0,70,0.4);
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #dd2476, #ff512f);
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(255,0,70,0.7);
}

/* Prediction Box */
.prediction {
    background: rgba(0, 0, 0, 0.6);
    border: 2px solid #00eaff;
    border-radius: 16px;
    padding: 22px;
    font-size: 22px;
    font-weight: bold;
    color: #00eaff;
    box-shadow: 0 0 20px rgba(0,234,255,0.6);
    backdrop-filter: blur(6px);
}

/* Titles */
h1, h2 {
    color: #00eaff;
    font-family: "Poppins", sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    text-shadow: 0 0 12px rgba(0, 234, 255, 0.4);
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Header ---
st.title("📰 Fake News Detection System")
st.subheader("Enter Article Details")

# --- Layout in two columns ---
col1, col2 = st.columns(2)

with col1:
    title_input = st.text_input("Title", value="")
    date_input = st.date_input("Date")
    source_input = st.selectbox("Source", ["CNN", "Fox News", "Reuters", "NY Times"], index=0)
    author_input = st.selectbox(
        "Author",
        ["Paula George", "Joseph Hill", "Julia Robinson", "David Foster DDS", "Austin Walker"],
        index=0,
    )

with col2:
    text_input = st.text_area("News Text", value="", height=150)
    category_input = st.selectbox("Category", ["Politics", "Business", "Science", "Technology"], index=0)
    subject_input = st.selectbox("Subject (optional)", ["", "World", "Health", "Sports"], index=0)

# --- Predict Button ---
if st.button("Predict"):
    if title_input.strip() == "" and text_input.strip() == "":
        st.warning("Please enter at least a title or news text.")
    else:
        full_text = title_input + " " + text_input

        year, month, day, dayofweek = (
            date_input.year,
            date_input.month,
            date_input.day,
            date_input.weekday(),
        )

        input_df = pd.DataFrame(
            {
                "full_text": [full_text],
                "year": [year],
                "month": [month],
                "day": [day],
                "dayofweek": [dayofweek],
                "source": [source_input],
                "author": [author_input],
                "category": [category_input],
                "subject": [subject_input if subject_input else ""],
            }
        )

        pred = model.predict(input_df)[0]
        label_map = {0: "Real", 1: "Fake"}

        st.markdown(
            f'<div class="prediction">The article is predicted as: <strong>{label_map[pred]}</strong></div>',
            unsafe_allow_html=True,
        )
