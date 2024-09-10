import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

def transform_text(text):
    """Preprocess the text by tokenizing, removing stopwords, punctuation, and stemming."""
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit application code
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the preprocessed text
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict using the loaded model
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
