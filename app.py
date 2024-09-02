import nltk
import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

# Ensure 'punkt' is downloaded
nltk.download('punkt')

# Initialize NLTK components
stemmer = PorterStemmer()
stopw = stopwords.words("english")
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

sms_input = st.text_area("Enter the SMS:")


def transform(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    text = word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopw and i not in string.punctuation]

    # Stem tokens
    text = [stemmer.stem(i) for i in text]

    # Join tokens back into a single string
    return " ".join(text)


if st.button("Predict"):
    # Transform and vectorize the input text
    transformed_sms = transform(sms_input)
    vectorized = tfidf.transform([transformed_sms])

    # Predict the class
    result = model.predict(vectorized)[0]

    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
