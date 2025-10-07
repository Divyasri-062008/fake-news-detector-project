# app.py
import streamlit as st
import pickle, requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load Model & Vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Summarizer Pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Prediction Function
def predict_news(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return pred, max(proba)

# Summarization Function
def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Streamlit App UI
st.title("📰 Fake News Detector for Students")
st.write("Easily check whether a news article or post is **Real or Fake**!")

option = st.radio("Choose Input Type:", ["Paste Text", "Upload File", "From URL"])

news_text = ""

if option == "Paste Text":
    news_text = st.text_area("Paste the news article here:")
elif option == "Upload File":
    uploaded = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded:
        news_text = uploaded.read().decode("utf-8")
elif option == "From URL":
    url = st.text_input("Enter news article URL:")
    if url:
        try:
            page = requests.get(url, timeout=10)
            soup = BeautifulSoup(page.content, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            news_text = " ".join(paragraphs)
        except Exception as e:
            st.error(f"Error fetching URL: {e}")

if st.button("🔎 Analyze"):
    if news_text.strip():
        with st.spinner("Analyzing..."):
            pred, confidence = predict_news(news_text)
            label = "✅ Real News" if pred == 1 else "❌ Fake News"

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}")

        with st.spinner("Summarizing..."):
            summary = summarize_text(news_text[:1000])  # limit to avoid long texts
            st.subheader("📝 Summary")
            st.write(summary)
    else:
        st.warning("Please enter or upload some text.")
