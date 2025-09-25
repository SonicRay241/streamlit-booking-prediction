# Hotel Booking Prediction with Streamlit

This project demonstrates a machine learning model that predicts whether a hotel booking will be **confirmed or canceled**. The model is trained using **XGBoost** and deployed with **Streamlit** as a simple, interactive web application.

## 🚀 Overview

The goal of this project is to predict the likelihood of a booking confirmation based on customer and reservation details.

Key features used for prediction include:

* Number of adults and children
* Length of stay (weekend nights, week nights)
* Meal plan and room type reserved
* Lead time before arrival
* Arrival year, month, and date
* Market segment type
* Repeated guest status
* Previous cancellations and non-cancellations
* Average price per room
* Special requests

The pipeline includes:

* Data preprocessing and feature engineering
* Model training with **XGBoost**
* Interactive prediction interface with **Streamlit**

## 🧰 Tech Stack

* **Python 3.9+**
* **XGBoost** – classification model for booking prediction
* **scikit-learn / Pandas / NumPy** – preprocessing and feature handling
* **Streamlit** – frontend web UI

## ✨ Features

* Predict hotel booking confirmation or cancellation in real-time
* Clean and interactive user interface for entering booking details
* High-performance model using **XGBoost**
* Easily extendable to new features or datasets

## 📈 Model Performance

The XGBoost model achieved strong performance on the test dataset:

* **Accuracy**: 0.87
* **Precision**: 0.79 (Canceled), 0.89 (Not Canceled)
* **Recall**: 0.71 (Canceled), 0.93 (Not Canceled)
* **F1-Score**: 0.75 (Canceled), 0.91 (Not Canceled)

Overall, the model demonstrates high predictive power, particularly in correctly identifying **non-canceled bookings** while maintaining solid performance on **canceled bookings**.

## 📄 License

This project is licensed under the MIT License.
