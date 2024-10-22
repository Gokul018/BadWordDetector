import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load your model and tokenizer
model = load_model('Model/bad_word_detector.h5')
with open('Model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 30  # Set this to your maximum sequence length
# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")
def predict_sentiment(paragraph):
    highlighted_text = []
    
    try:
        test_words = paragraph.split()
        
        # Tokenize the input words
        test_sequences = tokenizer.texts_to_sequences(test_words)
        
        # Pad the sequences
        test_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
        
        # Predict
        predictions = model.predict(test_pad)

        # Interpret Results
        for word, pred in zip(test_words, predictions):
            label = 1 if pred[0] >= 0.1 else 0  # Threshold for classification
            sentiment = "Bad" if label == 1 else "Good"

            # Highlight bad words
            if sentiment == "Bad":
                highlighted_text.append(f"<span style='color:red; font-weight:bold;'>{word}</span>")
            else:
                highlighted_text.append(word)

        # Join the words back into a single string
        return ' '.join(highlighted_text)
    
    except Exception as e:
        # Log the error for debugging (optional)
        print(f"Error: {e}")
        return "An error occurred while processing your input. Please try again."

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



with st.container():
    st.write("---")
    st.header("Error Occurred!")
    st.write("If you encountered an issue or any wrong predictions, please let us know:")
    
    # Contact form HTML
    contact_form = """
    <form action="https://formsubmit.co/grinstagram123@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" (optional)>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)