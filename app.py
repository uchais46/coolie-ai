
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Train the model inside the app (so we don't need a .pkl file)
stations = ['New Delhi', 'Mumbai CST', 'Howrah', 'Chennai Central', 'Bengaluru City']
seasons = ['Summer', 'Monsoon', 'Winter', 'Festival']
min_wages = {'New Delhi': 400, 'Mumbai CST': 450, 'Howrah': 350, 'Chennai Central': 370, 'Bengaluru City': 420}

# Generate dummy dataset
df = pd.DataFrame({
    'station': [stations[i % 5] for i in range(200)],
    'train_arrival_density': [i % 10 + 1 for i in range(200)],
    'season': [seasons[i % 4] for i in range(200)],
    'num_bags': [i % 5 + 1 for i in range(200)],
    'total_weight': [10 + (i % 90) for i in range(200)],
    'avg_bag_size': ['Small' if i % 3 == 0 else 'Medium' if i % 3 == 1 else 'Large' for i in range(200)],
    'platform_distance': [50 + (i % 450) for i in range(200)]
})

df['min_wage'] = df['station'].map(min_wages)
df['season_factor'] = df['season'].map({'Summer': 1.0, 'Monsoon': 1.1, 'Winter': 0.9, 'Festival': 1.3})
df['bag_size_factor'] = df['avg_bag_size'].map({'Small': 0.8, 'Medium': 1.0, 'Large': 1.2})

df['coolie_charge'] = (
    df['min_wage'] * 0.02 +
    df['total_weight'] * df['bag_size_factor'] * 0.5 +
    df['platform_distance'] * 0.1 +
    df['train_arrival_density'] * 5
) * df['season_factor']

df['coolie_charge'] = df['coolie_charge'].round(0)

X = df[['station', 'train_arrival_density', 'season', 'num_bags', 'total_weight',
        'avg_bag_size', 'platform_distance']]
y = df['coolie_charge']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['station', 'season', 'avg_bag_size'])],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Coolie AI", page_icon="🎒")
st.title("👦 Coolie AI 🚆")
st.markdown("### 💬 Let's find out how much a coolie (porter) might charge you at the railway station!")

stations_display = ['New Delhi', 'Mumbai CST', 'Howrah', 'Chennai Central', 'Bengaluru City']
seasons_display = ['Summer ☀️', 'Monsoon 🌧️', 'Winter ❄️', 'Festival 🎉']
bag_sizes_display = ['Small 👜', 'Medium 🎒', 'Large 🧳']

station = st.selectbox("📍 Which station are you at?", stations_display)
season = st.selectbox("🗓️ What season is it now?", seasons_display)
train_density = st.slider("🚂 How busy is the station (number of trains per hour)?", 1, 10, 5)
num_bags = st.slider("🎒 How many bags do you have?", 1, 5, 2)
total_weight = st.slider("⚖️ Total weight of your bags (kg)?", 10, 100, 30)
avg_bag_size = st.selectbox("📦 What size are your bags?", bag_sizes_display)
platform_distance = st.slider("🛤️ Distance to walk (in meters)?", 50, 500, 100)

# Clean labels
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
