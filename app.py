import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Load your model and tokenizer
model = load_model('Model/model.h5')

# Assuming you have saved your tokenizer, load it as well
# tokenizer = ... (load your tokenizer)

# Example tokenizer for demonstration
tokenizer = Tokenizer()
# Fit your tokenizer on your dataset (you should do this before running the app)
# tokenizer.fit_on_texts(...)

max_length = 20  # Set this to your maximum sequence length

def predict_sentiment(paragraph):
    words = paragraph.split()
    sequences = tokenizer.texts_to_sequences(words)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    predictions = model.predict(padded_sequences)
    
    highlighted_text = []
    for word, pred in zip(words, predictions):
        label = 1 if pred >= 0.1 else 0  # Threshold for classification
        sentiment = "Bad" if label == 1 else "Good"
        
        if sentiment == "Bad":
            highlighted_text.append(f"<span style='color:red; font-weight:bold;'>{word}</span>")
        else:
            highlighted_text.append(word)

    return ' '.join(highlighted_text)

# Streamlit UI
st.title("Bad Word Detector")
st.write("Enter a paragraph to check for bad words:")

user_input = st.text_area("Input Paragraph", "")

if st.button("Check"):
    if user_input:
        highlighted_output = predict_sentiment(user_input)
        st.markdown("**Highlighted Output:**")
        st.markdown(highlighted_output, unsafe_allow_html=True)
    else:
        st.warning("Please enter a paragraph to check.")
