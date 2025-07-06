import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model pipeline
model = joblib.load("coolie_model.pkl")

st.set_page_config(page_title="Coolie Helper AI", page_icon="🎒")
st.title("👦 Coolie Helper AI 🚆")
st.markdown("### 💬 Let's find out how much a coolie (porter) might charge you at the railway station!")

stations = ['New Delhi', 'Mumbai CST', 'Howrah', 'Chennai Central', 'Bengaluru City']
seasons = ['Summer ☀️', 'Monsoon 🌧️', 'Winter ❄️', 'Festival 🎉']
bag_sizes = ['Small 👜', 'Medium 🎒', 'Large 🧳']

station = st.selectbox("📍 Which station are you at?", stations)
season = st.selectbox("🗓️ What season is it now?", seasons)
train_density = st.slider("🚂 How busy is the station (number of trains per hour)?", 1, 10, 5)
num_bags = st.slider("🎒 How many bags do you have?", 1, 5, 2)
total_weight = st.slider("⚖️ Total weight of your bags (kg)?", 10, 100, 30)
avg_bag_size = st.selectbox("📦 What size are your bags?", bag_sizes)
platform_distance = st.slider("🛤️ Distance to walk (in meters)?", 50, 500, 100)

# Convert back to original categories for model input
season_clean = season.split()[0]
avg_bag_size_clean = avg_bag_size.split()[0]

if st.button("🔮 Predict Coolie Charge"):
    input_data = pd.DataFrame([{
        'station': station,
        'train_arrival_density': train_density,
        'season': season_clean,
        'num_bags': num_bags,
        'total_weight': total_weight,
        'avg_bag_size': avg_bag_size_clean,
        'platform_distance': platform_distance
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Coolie Charge: ₹{round(prediction)}")
    st.balloons()

    if st.button("✅ Yes, I'm okay with this price. Find a coolie!"):
        st.info("📡 Sending request to nearby coolies...")
        st.success("👷‍♂️ Coolie has accepted your request! He will meet you shortly at the platform.")
        st.markdown("---")
        st.markdown("Thank you for using **Coolie AI**! 🙏 Have a safe journey!")
