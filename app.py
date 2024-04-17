import numpy as np
from flask import Flask, render_template, request
import speech_recognition as sr
import tensorflow as tf

app = Flask(__name__)

loaded_model = tf.keras.models.load_model('toxicity_detection_model')

r = sr.Recognizer()

def transcribe_audio(audio, language="en"):
    try:
        text = r.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return None

def preprocess_text(text):
    tokens = text.split()
    tokens = [token.lower() for token in tokens]
    tokens = tokens[:100]
    padded_tokens = tokens + [''] * (100 - len(tokens))
    return padded_tokens

def get_prediction(sentence):
    processed_sentence = preprocess_text(sentence)  
    processed_sentence = np.array([processed_sentence])  
    processed_sentence = processed_sentence.ravel()  
    prediction = loaded_model.predict(processed_sentence)  
    return prediction[0][0]  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Result')
def result():
    return render_template('result.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return "No audio file uploaded"
    
    audio_file = request.files['audio_file']
    language = request.form.get('language', 'en')  # Get selected language or default to English
    
    with sr.AudioFile(audio_file) as source:
        audio = r.listen(source)
    
    transcribed_text = transcribe_audio(audio, language)
    
    if transcribed_text:
        prediction = get_prediction(transcribed_text)
        return render_template('result.html', transcribed_text=transcribed_text, prediction=prediction)
    else:
        return "Failed to transcribe audio"

if __name__ == '__main__':
    app.run(debug=True)
