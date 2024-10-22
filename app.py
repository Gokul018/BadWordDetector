import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
# Load your model and tokenizer
model = load_model('Model/bad_word_detector.h5')
# Load the tokenizer
with open('Model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
# Load your actual tokenizer here if saved
# tokenizer = Tokenizer()
# Uncomment and load your tokenizer if it’s saved
# with open('tokenizer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)

# Fit your tokenizer on your dataset if needed
# tokenizer.fit_on_texts(...)

max_length = 30  # Set this to your maximum sequence length

def predict_sentiment(paragraph):

    highlighted_text = []
    print(paragraph)
    test_words = paragraph.split()
    print(test_words)
    # 1. Preprocess the Test Words
    # Tokenize the input words
    test_sequences = tokenizer.texts_to_sequences(test_words)
    print(test_sequences)
    # Pad the sequences
    test_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    print(test_pad)
    # 2. Predict
    predictions = model.predict(test_pad)
    print(predictions)
    # 3. Interpret Results
    for word, pred in zip(test_words, predictions):
        label = 1 if pred >= 0.1 else 0  # Threshold for classification
        sentiment = "Bad" if label == 1 else "Good"
        print(f"Word: {word}, Prediction: {pred[0]:.4f}, Sentiment: {sentiment}")

        # Highlight bad words
        if sentiment == "Bad":
            highlighted_text.append(f"<span style='color:red; font-weight:bold;'>{word}</span>")
        else:
            highlighted_text.append(word)

    # Join the words back into a single string
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
