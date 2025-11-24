
import streamlit as st
import numpy as np
import pickle

LOAD_PATH = "/content/drive/MyDrive/SafeSwipeModel/"

model = pickle.load(open("Model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

THRESHOLD = 0.0007

st.set_page_config(page_title="SafeSwipe Fraud Transaction Detection System", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>SafeSwipe — Fraud Detection</h1>
    <p style='text-align: center; font-size:17px; color:#ddd;'>
        Enter transaction details below using sliders.
    </p>
    <hr style="opacity: 0.2;">
""", unsafe_allow_html=True)

st.markdown("### 🔧 **Transaction Inputs**")

user_inputs = {}

colA, colB = st.columns(2)

with colA:
    user_inputs["Time"] = st.slider("Time", 0.0, 172792.0, 50000.0, step=1000.0)

with colB:
    user_inputs["Amount"] = st.slider("Amount", 0.0, 5000.0, 50.0, step=1.0)



st.markdown("### 📊 **PCA Components (V1 – V28)**")

left_col, right_col = st.columns(2)

for i, col in enumerate(columns):
    if col in ["Time", "Amount"]:
        continue

    if i % 2 != 0:
        with left_col:
            user_inputs[col] = st.slider(col, -10.0, 10.0, 0.0, step=0.1)
    else:
        with right_col:
            user_inputs[col] = st.slider(col, -10.0, 10.0, 0.0, step=0.1)


st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Fraud", use_container_width=True)


def result_box(is_fraud, prob):

    if is_fraud:
        bg = "#ffe6e6"
        border = "#ff4d4d"
        title_color = "#cc0000"
        icon = "🚨"
        title = "FRAUD DETECTED"
    else:
        bg = "#e6ffed"
        border = "#33cc66"
        title_color = "#00994d"
        icon = "✅"
        title = "NOT FRAUD"

    html = f"""
<div style="background-color:{bg}; border-left:8px solid {border};
padding:25px 30px; border-radius:12px; margin-top:25px;
box-shadow:0 4px 12px rgba(0,0,0,0.12);">

<h2 style="color:{title_color}; margin:0; font-weight:700;">
{icon} {title}
</h2>

<p style="font-size:20px; margin-top:12px; color:#333;">
<b>Model Probability:</b> {prob:.4f}
</p>

<p style="font-size:15px; margin-top:8px; color:#555;">
Raw Probability Value: <b>{prob}</b>
</p>

</div>
"""

    st.markdown(html, unsafe_allow_html=True)


if predict_btn:

    arr = np.array([user_inputs[col] for col in columns]).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    prob = model.predict_proba(arr_scaled)[0][1]
    is_fraud = (prob >= THRESHOLD)

    result_box(is_fraud, prob)
