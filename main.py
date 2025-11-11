import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- App Title ---
st.title("📩 Spam Message Detector using Logistic Regression")

st.write("""
This Streamlit app trains a Logistic Regression model on your uploaded dataset  
to classify messages as **Spam** or **Not Spam**.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your 'spam.csv' file", type=["csv"])

if uploaded_file is not None:
    # --- Load and Clean Data ---
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    st.success("✅ Data Loaded Successfully!")
    st.dataframe(df.head())

    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )

    # --- TF-IDF Vectorization ---
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # --- Train Logistic Regression Model ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # --- Evaluation ---
    preds = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, preds)

    st.subheader("📊 Model Evaluation")
    st.write(f"**Accuracy:** {accuracy*100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, preds))

    # --- Confusion Matrix ---
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    st.pyplot(fig)

    # --- User Input for Prediction ---
    st.subheader("💬 Test a Custom Message")
    user_input = st.text_area("Enter a message to classify:")

    if st.button("Predict"):
        if user_input.strip():
            user_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(user_tfidf)[0]
            prob = model.predict_proba(user_tfidf)[0][1]

            if prediction == 'spam':
                st.error(f"🚨 This message is **SPAM** (Probability: {prob*100:.2f}%)")
            else:
                st.success(f"✅ This message is **NOT SPAM** (Probability: {(1-prob)*100:.2f}%)")
        else:
            st.warning("Please enter a message before predicting.")
else:
    st.info("⬆️ Upload a `.csv` file to get started.")
