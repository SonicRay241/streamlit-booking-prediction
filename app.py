import streamlit as st
import pandas as pd
import datetime
import time
import random
from model import Classifier

# load model
model = Classifier.load_model("models/oop_model.pkl")

st.set_page_config(
    page_title="Hotel Booking Prediction",
    page_icon=":hotel:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None
)

st.title("Hotel Booking Cancellation Prediction with XGBoost")

if "current_preset" not in st.session_state:
    st.session_state.no_of_adults = 0
    st.session_state.no_of_children = 0
    st.session_state.no_of_weekend_nights = 0
    st.session_state.no_of_week_nights = 0
    st.session_state.meal_plan = "Not Selected"
    st.session_state.required_parking_space = False
    st.session_state.room_type = "Type 1"
    st.session_state.lead_time = 0
    st.session_state.arrival_date = datetime.date(2017,1,1)
    st.session_state.market_segment = "Offline"
    st.session_state.repeated_guest = False
    st.session_state.no_of_previous_cancellations = 0
    st.session_state.no_of_previous_booking_not_canceled = 0
    st.session_state.avg_price_per_room = 200.0
    st.session_state.no_of_special_requests = 0
    st.session_state.current_preset = 0

with st.sidebar:
    st.title("Presets")

    if st.button("Not Canceled Preset"):
        st.session_state.no_of_adults = 4
        st.session_state.no_of_children = 3
        st.session_state.no_of_weekend_nights = 2
        st.session_state.no_of_week_nights = 0
        st.session_state.meal_plan = "Meal Plan 3"
        st.session_state.required_parking_space = False
        st.session_state.room_type = "Type 2"
        st.session_state.lead_time = 261
        st.session_state.arrival_date = datetime.date(2018,12,23)
        st.session_state.market_segment = "Online"
        st.session_state.repeated_guest = True
        st.session_state.no_of_previous_cancellations = 0
        st.session_state.no_of_previous_booking_not_canceled = 2
        st.session_state.avg_price_per_room = 200.0
        st.session_state.no_of_special_requests = 1
        st.session_state.current_preset = 1

    if st.button("Canceled Preset"):
        st.session_state.no_of_adults = 2
        st.session_state.no_of_children = 0
        st.session_state.no_of_weekend_nights = 0
        st.session_state.no_of_week_nights = 3
        st.session_state.meal_plan = "Meal Plan 1"
        st.session_state.required_parking_space = False
        st.session_state.room_type = "Type 1"
        st.session_state.lead_time = 160
        st.session_state.arrival_date = datetime.date(2017,9,4)
        st.session_state.market_segment = "Corporate"
        st.session_state.repeated_guest = True
        st.session_state.no_of_previous_cancellations = 0
        st.session_state.no_of_previous_booking_not_canceled = 7
        st.session_state.avg_price_per_room = 150.0
        st.session_state.no_of_special_requests = 0
        st.session_state.current_preset = 2

no_of_adults = st.slider("Number of Adults", 0, 10, 1, key="no_of_adults")
no_of_children = st.slider("Number of Children", 0, 15, 1, key="no_of_children")
no_of_weekend_nights = st.slider("Number of Weekend Nights", 0, 15, 1, key="no_of_weekend_nights")
no_of_week_nights = st.slider("Number of Week Nights", 0, 30, 1, key="no_of_week_nights")
meal_plan = st.selectbox("Meal Plan", options=["Not Selected", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"], key="meal_plan")
required_parking_space = st.checkbox("Parking Space", key="required_parking_space") # Change to 0 and 1 later
room_type = st.select_slider("Room Type", options=["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6", "Type 7"], key="room_type")
lead_time = st.number_input("Lead Time", min_value=0, max_value=730, step=1, key="lead_time")
arrival_date = st.date_input("Arrival Date", min_value=datetime.date(2017, 1, 1), max_value=datetime.date(2018, 12, 31), key="arrival_date")
market_segment = st.selectbox("Market Segment", options=["Aviation", "Complementary", "Corporate", "Offline", "Online"], key="market_segment")
repeated_guest = st.checkbox("Repeated Guest", key="repeated_guest") # Change to 0 and 1 later
no_of_previous_cancellations = st.slider("Number of Previous Cancellations", 0, 15, 1, key="no_of_previous_cancellations")
no_of_previous_bookings_not_canceled = st.slider("Number of Previous Bookings Not Cancelled", 0, 60, 1, key="no_of_previous_booking_not_canceled")
avg_price_per_room = st.number_input("Average Price per Room (in Euros)", min_value=0.0, max_value=600.0, step=10.0, key="avg_price_per_room")
no_of_special_requests = st.slider("Number of Special Requests", 0, 10, 1, key="no_of_special_request")

if st.button("Predict", type="primary"):
    df = pd.DataFrame([[
            no_of_adults,
            no_of_children,
            no_of_weekend_nights,
            no_of_week_nights,
            meal_plan,
            1 if required_parking_space else 0,
            "Room_" + room_type,
            lead_time,
            arrival_date.year,
            arrival_date.month,
            arrival_date.day,
            market_segment,
            1 if repeated_guest else 0,
            no_of_previous_cancellations,
            no_of_previous_bookings_not_canceled,
            avg_price_per_room,
            no_of_special_requests
        ]],
        columns=["no_of_adults", "no_of_children", "no_of_weekend_nights",
        "no_of_week_nights", "type_of_meal_plan", "required_car_parking_space",
        "room_type_reserved", "lead_time", "arrival_year", "arrival_month",
        "arrival_date", "market_segment_type", "repeated_guest",
        "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
        "avg_price_per_room", "no_of_special_requests"]
    )

    with st.spinner("🧠 Thinking...", show_time=True):
        time.sleep(random.uniform(0.5, 3.0))

    pred = model.predict(df)[0]
        
    if pred == "Not_Canceled":
        st.success(f"Prediction: {pred.replace('_', ' ')}", icon="✔")
    else:
        st.error(f"Prediction: {pred.replace('_', ' ')}", icon="❌")
