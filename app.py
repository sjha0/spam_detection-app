import streamlit as st
import pandas as pd
import joblib  # For loading the model

# Load trained spam detection model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📧 Email Spam Detection App")

# Tab layout for single message or CSV file processing
tab1, tab2 = st.tabs(["📩 Single Message", "📂 Bulk CSV File"])

# ------------------------ SINGLE EMAIL DETECTION ------------------------
with tab1:
    st.subheader("🔍 Check a Single Email")
    
    # Input text box for a single email message
    user_input = st.text_area("✍️ Enter an email message:", height=150)
    
    if st.button("Detect Spam"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            # Convert text to vector
            input_vector = vectorizer.transform([user_input])
            
            # Predict spam/ham
            prediction = model.predict(input_vector)[0]
            result = "🚨 Spam!" if prediction == 1 else "✅ Not Spam (Ham)"
            
            # Display result
            st.success(f"**Prediction:** {result}")

# ------------------------ BULK CSV FILE DETECTION ------------------------
with tab2:
    st.subheader("📂 Upload a CSV File for Bulk Prediction")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the CSV
        st.write("📊 **CSV File Preview:**")  
        st.dataframe(df.head())

        # Check if the necessary column is present
        if "text" not in df.columns:
            st.error("❌ CSV file must contain a 'text' column with email content.")
        else:
            # Convert text data into numerical features using the vectorizer
            X_test = vectorizer.transform(df["text"])

            # Predict spam/ham using the trained model
            predictions = model.predict(X_test)

            # Add predictions to the DataFrame
            df["Prediction"] = ["🚨 Spam" if label == 1 else "✅ Not Spam" for label in predictions]

            # Show results
            st.write("✅ **Prediction Results:**")
            st.dataframe(df[["text", "Prediction"]])

            # Download processed file
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="spam_predictions.csv",
                mime="text/csv",
            )

