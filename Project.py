import streamlit as st
from streamlit_extras.let_it_rain import rain
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

fileName = "finalized_model.pkl"
with open(fileName, 'rb') as file:
    accuracy = pickle.load(file)
    model = pickle.load(file)
    stdScaler = pickle.load(file)
    pca = pickle.load(file)

LabelEncoders = {"Male":1, "Female":0, "Yes":1, "No":0, "Normal":1, "Fixed Defect":2, "Reversible Defect":3,
                 "Upward Slope":0, "Flat Slope":1, "Downward Slope":2}

st.set_page_config(page_title="Heart Checkup")
col1, col2 = st.columns([5, 1])
with col1:
    st.title("Heart Disease Risk Checkup")
with col2:
    st.image("images/heart.png", width=75)
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### Fill The Following Form for Diagnosis")
with col2:
    st.image("images/health-check.png", width=50)
st.markdown("---")

col1, col2= st.columns(2)

with col1:
    Sex = st.radio("Gender", ["Male", "Female"])
with col2:
    Exang = st.radio("Chest Pain Triggered By  Exercise", ["Yes", "No"])

BloodFlow = st.radio("Blood Flow", ["Normal", "Fixed Defect", "Reversible Defect"])
STSlope = st.radio("ST Slope", ["Upward Slope", "Flat Slope", "Downward Slope"])

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=1, max_value=100, value=50)
with col2:
    BloodPressure = st.number_input("Blood Pressure at Rest", min_value=50, max_value=250, value=120)

col1, col2 = st.columns(2)

with col1:
    HeartRate = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=80)
with col2:
    Cholestrol = st.number_input("Cholestrol", min_value=50, max_value=600, value=150)
oldPeak = st.number_input("ST segment drop during exercise", min_value=0.0, max_value=10.0, step=0.1)

ChestPain = st.slider("Chest Pain Strength (No Pain → Severe Pain)",min_value=0, max_value=3)
MajorVessels = st.slider("Number OF Involved Major Vessels",min_value=0, max_value=3)

submitButton = st.button("Check", use_container_width=True)
st.markdown("---")

if(submitButton):
    X = [Age, BloodPressure, Cholestrol, HeartRate, oldPeak, Sex, ChestPain, Exang,  STSlope, MajorVessels, BloodFlow]
    X[5:] = [LabelEncoders[i] if isinstance(i, str) else i for i in X[5:]]
    X = list(map(int, X))
    X = np.array(X).astype(float).reshape(1, -1)

    X[:, :5] = stdScaler.transform(X[:, :5])
    X[:, :5] = pca.transform(X[:, :5])[0]

    Prediction = model.predict(X)

    col1, col2 = st.columns([1.8, 1])
    img, text, emoji, color = None, None, None, None
    tips = '''
        Tips To Keep Your Heart Healthy
        Eat a Balanced Diet 🥗
        Stay Active 🏃‍♂️
        Maintain a Healthy Weight ⚖️
        Monitor Your Blood Pressure, Cholesterol, and Blood Sugar 🩺
        Avoid Smoking and Alcohol 🚭🍷"
        '''
    if Prediction == 0:
        img = "images/health.png"
        text = "You Don't Have a Heart Issue"
        color = "#88E788"
        emoji = "✨"
    else:
        img = "images/heart-attack.png"
        text = "You May Have a Heart Issue"
        color = "#FF0000"
        emoji = "🚑"
    with col1:
        st.markdown(f'<span style="color:{color}; font-weight:bold; font-size:20px;">{text}</span>', unsafe_allow_html=True)
        st.text(tips)
    with col2:
        st.image(img)
    rain(emoji=emoji, font_size=20, falling_speed=4, animation_length=5)
    st.markdown(f'''
    <span style="color:red; font-size:14px;">
    ⚠️ This model has an accuracy of {accuracy:.2%}. The results are for informational purposes only and do not replace professional medical advice. Please consult a healthcare provider for any concerns.
    </span>
    ''', unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Make button container a flex container */
    div[data-testid="stRadio"] > div{
        display: flex;
        flex-direction: row;
        justify-content: center;
    }
    /* Make the actual button stretch */
    div[data-testid="stButton"] > button:first-child{
        width: 100%;                 /* fill available space */
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    div[data-testid="stButton"] > button:first-child:hover {
        background-color: #45a049;
        font-size: 20px;
    }
    div[data-testid="stButton"] > button:first-child:active {
        background-color :green;
    }
    </style>
""", unsafe_allow_html=True)