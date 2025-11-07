from nltk.tokenize import word_tokenize
import contractions
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

stopwords_en=stopwords.words('english')


negative_words = {"no", "not", "never", "none", "n't"}
stopwords_en=set(stopwords_en)

custom_stop_words = stopwords_en - negative_words


lemmatizer=WordNetLemmatizer()

def clean_text(text):

    text = contractions.fix(text)

    # Remove punctuation, symbols, numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Lowercase
    text = text.lower()

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords & short tokens
    filtered = [w for w in words if w not in custom_stop_words and len(w) > 1]

    # Lemmatize
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

    # Rejoin and normalize spaces
    cleaned = " ".join(lemmatized)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

classes_names = ['Positive','Neutral','Negative','Irrelevant']

def get_pred(model,tok,text,max_len):
    
    seq = tok.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    
    # Predict
    pred_prob = model.predict(padded)
    pred_class = np.argmax(pred_prob, axis=1)[0] 
    class_name = classes_names[pred_class]
    
    return class_name
    