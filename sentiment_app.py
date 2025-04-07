
import streamlit as st
import joblib

# Load the TF-IDF vectorizer and sentiment model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

# Title
st.title("🇦🇺 Australia Wikipedia Sentiment Predictor")

# Input from user
user_input = st.text_area("Enter a sentence about Australia:")

# Predict sentiment
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        # Display result
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Predicted Sentiment: {sentiment}")
