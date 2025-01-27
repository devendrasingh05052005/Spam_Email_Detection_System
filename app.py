import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
    }
    .spam-result {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# Load the models
@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# Load models
tfidf, model = load_models()

# Main app
st.title("📧 Advanced Email/SMS Spam Detector")
st.markdown("---")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Message Analysis")
    input_sms = st.text_area("Enter the message to analyze", height=150)

    if st.button('Analyze Message'):
        if input_sms:
            # Add a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Processing animation
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Analyzing... {i + 1}%")
                time.sleep(0.01)

            # Transform and predict
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            prediction_prob = model.predict_proba(vector_input)[0]

            # Display result with custom styling
            if result == 1:
                st.markdown(
                    f"""
                    <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border: 1px solid #ffcdd2;">
                        <h2 style="color: #c62828;">⚠️ Spam Detected!</h2>
                        <p>This message has been identified as spam with {prediction_prob[1] * 100:.2f}% confidence.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border: 1px solid #c8e6c9;">
                        <h2 style="color: #2e7d32;">✅ Not Spam</h2>
                        <p>This message appears to be legitimate with {prediction_prob[0] * 100:.2f}% confidence.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Message Statistics
            st.markdown("### Message Statistics")
            col_stats1, col_stats2, col_stats3 = st.columns(3)

            with col_stats1:
                st.metric("Message Length", len(input_sms))
            with col_stats2:
                st.metric("Word Count", len(input_sms.split()))
            with col_stats3:
                st.metric("Unique Words", len(set(input_sms.split())))

with col2:
    st.markdown("### How it Works")
    st.markdown("""
    1. **Text Preprocessing** 📝
       - Converts to lowercase
       - Removes special characters
       - Removes stop words

    2. **Feature Extraction** 🔍
       - Vectorizes the text
       - Analyzes patterns

    3. **AI Analysis** 🤖
       - Uses machine learning
       - Predicts spam probability
    """)

    # Sample spam indicators
    st.markdown("### Common Spam Indicators")
    spam_indicators = {
        'FREE': 85,
        'Winner': 78,
        'Prize': 72,
        'Urgent': 65,
        'Limited time': 60
    }

    # Create a bar chart using plotly
    df = pd.DataFrame(list(spam_indicators.items()), columns=['Indicator', 'Spam Probability'])
    fig = px.bar(df, x='Indicator', y='Spam Probability',
                 color='Spam Probability',
                 color_continuous_scale=['green', 'yellow', 'red'])
    fig.update_layout(
        title="Spam Probability by Keywords",
        xaxis_title="Keywords",
        yaxis_title="Spam Probability (%)"
    )
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🛡️ Advanced Spam Detection System | Made with  by Devendra singh</p>
</div>
""", unsafe_allow_html=True)