import streamlit as st
import requests

st.title("AI Model Prediction")

st.write("Enter 4 numbers (comma-separated):")
st.write("0 → Iris-setosa
1 → Iris-versicolor
2 → Iris-virginica")
input_data = st.text_input("Example: 5.1, 3.5, 1.4, 0.2")

if st.button("Predict"):
    try:
        # Convert user input to a list of 4 floats
        input_list = [float(x.strip()) for x in input_data.split(",")]

        # Check if exactly 4 features are provided
        if len(input_list) != 4:
            st.error("Error: Please enter exactly 4 values.")
        else:
            # Send request to Flask API (Ensure this is the correct endpoint)
            api_url = "https://fast-api-testing-w0l1.onrender.com/predict"
            response = requests.post(api_url, json={"features": input_list})

            # Check for a successful response
            if response.status_code == 200:
                prediction = response.json().get("prediction", "No prediction found")
                st.success(f"Prediction: {prediction}")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

    except ValueError:
        st.error("Error: Please enter numbers separated by commas.")
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {e}")
