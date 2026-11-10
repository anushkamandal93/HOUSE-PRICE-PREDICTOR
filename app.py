import json
import os

import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "house_model.joblib")
META_PATH = os.path.join(BASE, "model", "meta.json")

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python train_model.py` first, then refresh this page.")
    st.stop()

model, meta = load_model()

OCEAN_LABELS = {
    "<1H OCEAN": "Less than 1 hour from ocean",
    "NEAR OCEAN": "Near ocean",
    "NEAR BAY": "Near bay",
    "INLAND": "Inland",
    "ISLAND": "Island",
}

# city: (latitude, longitude, ocean_proximity)
CITIES = {
    "Custom location": None,
    "San Francisco": (37.77, -122.42, "NEAR BAY"),
    "San Jose": (37.34, -121.89, "<1H OCEAN"),
    "Los Angeles": (34.05, -118.24, "<1H OCEAN"),
    "San Diego": (32.72, -117.16, "NEAR OCEAN"),
    "Sacramento": (38.58, -121.49, "INLAND"),
    "Fresno": (36.74, -119.79, "INLAND"),
}

DEFAULTS = {
    "city": "Custom location",
    "lat": 34.05,
    "lon": -118.24,
    "ocean": "<1H OCEAN",
    "income": 3.9,
    "age": 29,
    "rooms": 2600,
    "bedrooms": 540,
    "population": 1400,
    "households": 500,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def on_city_change():
    place = CITIES[st.session_state.city]
    if place:
        st.session_state.lat, st.session_state.lon, st.session_state.ocean = place


def reset_form():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# ---------------------------------------------------------------- header
st.title("🏠 California House Price Predictor")
st.write(
    f"Describe a district and a Random Forest model trained on {meta['n_rows']:,} "
    "California districts estimates its median house value."
)

left, right = st.columns([3, 2], gap="large")

# ---------------------------------------------------------------- inputs
with left:
    st.subheader("District details")

    st.markdown("**Location**")
    st.selectbox("Quick pick a city", list(CITIES), key="city", on_change=on_city_change)
    c1, c2 = st.columns(2)
    c1.number_input("Latitude", min_value=32.54, max_value=41.95, step=0.01, format="%.2f", key="lat")
    c2.number_input("Longitude", min_value=-124.35, max_value=-114.31, step=0.01, format="%.2f", key="lon")
    st.selectbox(
        "Distance from the ocean",
        list(OCEAN_LABELS),
        format_func=OCEAN_LABELS.get,
        key="ocean",
    )

    st.markdown("**Neighbourhood**")
    st.slider(
        "Median income (in $10,000s)",
        min_value=0.5, max_value=15.0, step=0.1, key="income",
        help="3.9 means about $39,000 a year.",
    )
    st.caption(f"= ${st.session_state.income * 10000:,.0f} a year")
    st.slider("Median house age (years)", min_value=1, max_value=52, key="age")

    st.markdown("**Size of the district**")
    d1, d2 = st.columns(2)
    d1.number_input("Total rooms", min_value=1, max_value=40000, step=50, key="rooms")
    d2.number_input("Total bedrooms", min_value=1, max_value=7000, step=10, key="bedrooms")
    d3, d4 = st.columns(2)
    d3.number_input("Population", min_value=1, max_value=40000, step=50, key="population")
    d4.number_input("Households", min_value=1, max_value=7000, step=10, key="households")

    st.button("Reset", on_click=reset_form)

# ---------------------------------------------------------------- result
with right:
    st.subheader("Estimated median house value")

    s = st.session_state
    problem = None
    if s.bedrooms > s.rooms:
        problem = "Total bedrooms cannot be more than total rooms."
    elif s.households > s.population:
        problem = "Households cannot be more than population."

    if problem:
        st.error(problem)
    else:
        row = {
            "longitude": s.lon,
            "latitude": s.lat,
            "housing_median_age": s.age,
            "total_rooms": s.rooms,
            "total_bedrooms": s.bedrooms,
            "population": s.population,
            "households": s.households,
            "median_income": s.income,
            "ocean_proximity": s.ocean,
        }
        # same column order as in training
        frame = pd.DataFrame([row])[meta["columns"]]
        price = float(model.predict(frame)[0])
        low = max(0, price - meta["rmse"])
        high = price + meta["rmse"]

        st.metric("Prediction", f"${price:,.0f}")
        st.write(f"Likely range: **${low:,.0f}** to **${high:,.0f}**")
        if price >= 480000:
            st.warning("This is close to the $500,000 cap in the data, so the real value could be higher.")

    m1, m2 = st.columns(2)
    m1.metric("Typical error (RMSE)", f"±${meta['rmse']:,}")
    m2.metric("R² on unseen data", meta["r2"])

    st.map(pd.DataFrame({"lat": [s.lat], "lon": [s.lon]}), zoom=5)

    st.caption(
        "The data is from the 1990 US census, so prices are 1990-era values, not today's market. "
        "Use this as a learning project, not for real decisions."
    )
