from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import gradio as gr

# Load the pre-trained model and tokenizer
model = load_model("model.keras")
tokenizer = joblib.load("tokenizer.pkl")

# Define the predictive system
def predictive_system(review):
    if not review.strip():
        return "empty review"
    elif review.strip().isnumeric():
        return "review contains only numbers"
    elif all(char in "!@#$%^&*()_+-=[]{}|;:',.<>?/`~ " for char in review):
        return "review contains only symbols"
    elif len(review.split()) == 1:
        return "review is too short"
    elif not re.search(r"[a-zA-Z]", review):
        return "review is gibberish"
    else:
        sequences = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequences, maxlen=200)
        prediction = model.predict(padded_sequence)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
        return sentiment

# Gradio Interface
title = "Movie Sentiment Analysis Using Machine Learning"
article = "<div style='text-align: center; padding-top: 50px;'><p>Created by <strong>Vipanshu Suman & Divyam Pathak</strong></p></div>"

app = gr.Interface(
    fn=predictive_system, 
    inputs="textbox", 
    outputs="textbox", 
    title=title, 
    article=article
)

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)
