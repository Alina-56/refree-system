import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Streamlit app
st.title("AI Referee System")
st.write("Predict match outcomes (Win/Not Win) based on Expected Goals (xG) and Shots.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Data preprocessing
    try:
        data = data[['xg', 'sh', 'result']]
        data['result_binary'] = data['result'].apply(lambda x: 1 if x == 'W' else 0)
        data = data.drop(columns=['result'])

        # Split data into features and target
        X = data[['xg', 'sh']]
        y = data['result_binary']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_class = (y_pred > 0.5).astype(int)  # Convert probabilities to binary classes
        accuracy = accuracy_score(y_test, y_pred_class)

        # Sidebar inputs
        st.sidebar.header("Input Features")
        xg = st.sidebar.slider("Expected Goals (xG)", float(X['xg'].min()), float(X['xg'].max()), float(X['xg'].mean()))
        sh = st.sidebar.slider("Shots", int(X['sh'].min()), int(X['sh'].max()), int(X['sh'].mean()))

        # Prediction for user input
        input_data = np.array([[xg, sh]])
        prediction = model.predict(input_data)
        prediction_class = "Win" if prediction > 0.5 else "Not Win"

        st.write(f"Prediction: **{prediction_class}**")
        st.write(f"Model Accuracy: {accuracy:.2f}")
    except KeyError:
        st.error("The uploaded file does not have the required columns: 'xg', 'sh', 'result'.")
else:
    st.write("Please upload a CSV file to proceed.")
