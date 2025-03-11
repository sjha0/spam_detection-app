import pandas as pd
from tkinter import Tk, filedialog

# Open file dialog to select the CSV file
Tk().withdraw()  # Hide the root window
import streamlit as st
import pandas as pd

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Read CSV file
    st.write("📊 **CSV File Preview:**")  
    st.dataframe(df)  # Display the CSV file in a table


# Load the CSV file
df = pd.read_csv("C:\\Users\\Saurabh Jha\\Downloads\\spam.csv", encoding="latin-1")

# Print first few rows
print(df.head())
print(df.head())  # Display the first few rows
print(df.columns)  # Check actual column names

df = df[['label', 'text']]  # Adjust based on actual column names
df.columns = ['label', 'message']  # Standardized names
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['processed_message'] = df['message'].apply(preprocess_text)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_message']).toarray()
y = df['label']
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
def predict_spam(email):
    email = preprocess_text(email)
    email_vector = vectorizer.transform([email]).toarray()
    prediction = model.predict(email_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example tests
print(predict_spam("Congratulations! You won a lottery. Claim now!"))
print(predict_spam("Hey, let's meet for coffee tomorrow."))
import joblib

# Save the trained model
joblib.dump(model, "spam_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
import joblib

# Load the saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocess function (same as before)
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Function to predict if an email is spam or not
def predict_spam(email):
    email = preprocess_text(email)
    email_vector = vectorizer.transform([email]).toarray()
    prediction = model.predict(email_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example Predictions
print(predict_spam("Congratulations! You won a lottery. Claim now!"))
print(predict_spam("Hey, let's meet for coffee tomorrow."))
