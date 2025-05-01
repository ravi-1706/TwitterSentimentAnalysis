import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# Download stopwords once
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Initialize Nitter scraper
@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# Sentiment prediction logic
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = ' '.join([word for word in text if word not in stop_words])
    text_vector = vectorizer.transform([text])
    sentiment = model.predict(text_vector)
    return "Negative" if sentiment == 0 else "Positive"

# Modern tweet sentiment card
def create_card(tweet_text, sentiment):
    color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
    emoji = "😊" if sentiment == "Positive" else "😠"
    card_html = f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 15px;
                margin: 15px 0; box-shadow: 0 6px 12px rgba(0,0,0,0.15); transition: 0.3s;">
        <h4 style="color: white; margin-bottom: 10px;">{emoji} {sentiment} Sentiment</h4>
        <p style="color: white; font-size: 16px; line-height: 1.5;">{tweet_text}</p>
    </div>
    """
    return card_html

# Custom CSS
def add_custom_css():
    st.markdown("""
        <style>
            .css-1v0mbdj, .css-ffhzg2 {padding-top: 1rem !important;}
            textarea, input {
                border-radius: 8px !important;
            }
            .stButton button {
                background-color: #1abc9c;
                color: white;
                border-radius: 10px;
                padding: 0.5rem 1.2rem;
                font-weight: 600;
                transition: 0.3s;
            }
            .stButton button:hover {
                background-color: #16a085;
                transform: scale(1.05);
            }
        </style>
    """, unsafe_allow_html=True)

# Main app
def main():
    st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")
    add_custom_css()

    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>📊 Twitter Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: gray;'>Analyze tweets and text sentiment instantly!</p>", unsafe_allow_html=True)
    st.markdown("---")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    option = st.radio("Choose an option", ["✍️ Input text", "🐦 Get tweets from username"], horizontal=True)

    if option == "✍️ Input text":
        with st.container():
            text_input = st.text_area("Enter your text below", height=150, placeholder="Type your text here...")
            if st.button("🔍 Analyze Sentiment"):
                if text_input.strip():
                    sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                    st.success(f"**Sentiment:** {sentiment}")
                else:
                    st.warning("Please enter some text.")

    elif option == "🐦 Get tweets from username":
        username = st.text_input("Enter Twitter username (without @):", placeholder="e.g., elonmusk")
        if st.button("📥 Fetch Tweets"):
            tweets_data = scraper.get_tweets(username, mode='user', number=5)
            if 'tweets' in tweets_data:
                st.subheader(f"Latest Tweets from @{username}")
                for tweet in tweets_data['tweets']:
                    tweet_text = tweet['text']
                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                    st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
            else:
                st.error("Could not retrieve tweets. Please check the username or try again later.")

    st.markdown("<hr style='margin-top: 40px;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
