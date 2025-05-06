import streamlit as st
import joblib

# Load model and vectorizer from specified paths
MODEL_PATH = r'E:\ML project\spam_classifier_model.pkl'
VECTORIZER_PATH = r'E:\ML project\tfidf_vectorizer.pkl'

@st.cache_resource
def load_components():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

model, vectorizer = load_components()

def predict_email(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    return "Spam" if pred == 1 else "Ham"

# Streamlit UI
st.title("📧 Email Spam Detector")
st.write("This app predicts if an email is spam or ham using a pre-trained ML model.")

email_text = st.text_area("Paste your email text here:", height=200)

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter some email text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            prediction = predict_email(email_text)
            color = "#FF0000" if prediction == "Spam" else "#00FF00"
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color}20; margin: 20px 0;">
                <h3 style="color: {color}; margin:0;">Prediction: {prediction}</h3>
            </div>
            """, unsafe_allow_html=True)

# Optional: Add model info section
with st.expander("ℹ️ Model Information"):
    st.write("""
    **Model Details:**
    - Algorithm: Naive Bayes Classifier
    - Vectorizer: TF-IDF
    - Training Data: Spam/Ham dataset
    - Last Updated: [Add your model date]
    """)