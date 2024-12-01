import streamlit as st
import pandas as pd
from joblib import load

# Load the model
MODEL_PATH = "model.joblib"

@st.cache_resource
def load_model():
    try:
        return load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Predict", "Link Prediction", "EDA"])

# Load the model
model = load_model()

# Page 1: Upload & Predict
if page == "Upload & Predict":
    st.title("Upload Dataset & Predict")

    uploaded_file = st.file_uploader("Upload a CSV file for predictions", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)

        if model is not None:
            try:
                predictions = model.predict(data)
                data["Predictions"] = predictions
                st.write("Predictions:")
                st.dataframe(data)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Model not loaded. Please check your file.")

# Page 2: Link Prediction
elif page == "Link Prediction":
    st.title("Phishing Link Prediction")

    # Input URL from the user
    url_input = st.text_input("Enter a link to check if it's phishing or not:")

    if url_input:
        # Preprocessing the input URL
        # Assuming the model expects certain feature transformations, you can add steps here
        # For simplicity, let's just use the raw URL in this example
        try:
            # Preprocess the URL (adjust this depending on model requirements)
            # Example: convert the URL into features (e.g., using a vectorizer or custom function)
            # Here, we assume the model expects the raw URL or some form of encoded features
            input_features = [url_input]  # This should be replaced with actual feature extraction

            # Make a prediction
            prediction = model.predict(input_features)

            # Display the result
            if prediction == 1:
                st.write(f"The URL: **{url_input}** is **Phishing**.")
            else:
                st.write(f"The URL: **{url_input}** is **Not Phishing**.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Page 3: EDA
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)

        # Show basic stats
        st.write("Basic Statistics:")
        st.write(data.describe())

        # Show missing values
        st.write("Missing Values:")
        st.write(data.isnull().sum())

        # Correlation heatmap (if numeric data available)
        if len(data.select_dtypes(include=["number"]).columns) > 1:
            st.write("Correlation Heatmap:")
            st.area_chart(data.corr())

# Footer
st.sidebar.info("Developed by [Rezha Novandra / 22191259]")
