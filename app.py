from flask import Flask, request, render_template
import pickle


import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("punkt")
import string
import re
lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words("english"))
def preprocessing(text:str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = "".join(char for char in text if char not in string.punctuation)
    tokens = text.split()
    clean_text = [lemmatizer.lemmatize(word=word) for word in tokens if word not in sw]
    return " ".join(clean_text)


app = Flask(__name__)


rf_model = pickle.load('components\\classification_model.pkl')
vectorizer = pickle.load('components\\vectorizer.pkl.pkl')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_text = request.form['text']
    
    # Preprocess (same as training)
    processed_text = preprocess_function(user_text)
    
    # Feature extraction
    features = vectorizer.transform([processed_text])
    
    # Predictions
    fake_pred = rf_model.predict(features)[0]
    fake_prob = rf_model.predict_proba(features)[0].max()
        
    return {
        'prediction': 'FAKE' if fake_pred == 1 else 'REAL',
        'confidence': fake_prob * 100,
    }