import streamlit as st
import requests

st.title("AI Model Prediction")

st.write("Enter 4 numbers (comma-separated):")
input_data = st.text_input("Example: 5.1, 3.5, 1.4, 0.2")

if st.button("Predict"):
    try:
        # Convert user input to a list of 4 floats
        input_list = [float(x) for x in input_data.split(",")]

        # Check if exactly 4 features are provided
        if len(input_list) != 4:
            st.write("Error: Please enter exactly 4 values.")
        else:
            # Send request to Flask API
            response = requests.post("https://fast-api-testing-w0l1.onrender.com", json={"features": input_list})

            # Debugging: Print the full response to see if it's correct
            st.write("Raw API Response:", response.text)

            prediction = response.json().get("prediction", "No prediction found")
            st.write("Prediction:", prediction)

    except ValueError:
        st.write("Error: Please enter numbers separated by commas.")
