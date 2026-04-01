import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Marketing Conversion Predictor", layout="wide")

st.title("📊 Marketing Conversion Prediction App")
st.write("Upload your dataset and get conversion predictions instantly.")

# =========================
# LOAD MODEL SAFELY
# =========================
MODEL_PATH = "dm_conversion_best_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please check GitHub upload.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Uploaded Data")
        st.dataframe(df.head())

        # =========================
        # PREDICTION
        # =========================
        predictions = model.predict(df)

        # If probability available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1]
            df["Conversion_Probability"] = probs

        df["Prediction"] = predictions

        # =========================
        # OUTPUT
        # =========================
        st.subheader("✅ Predictions")
        st.dataframe(df)

        # =========================
        # DOWNLOAD BUTTON
        # =========================
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

else:
    st.info("👆 Please upload a CSV file to get predictions.")
