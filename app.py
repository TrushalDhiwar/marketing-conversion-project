import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Marketing Conversion Predictor")

st.title("📊 Marketing Conversion Prediction App")

st.write("This app predicts whether a user will convert based on input data.")

# Load model
model = joblib.load("dm_conversion_best_model.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📂 Uploaded Data")
        st.dataframe(df.head())

        # Prediction
        predictions = model.predict(df)

        df["Prediction"] = predictions

        st.subheader("✅ Predictions")
        st.dataframe(df)

        # Download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")