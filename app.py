import logging
from flask import Flask, request, render_template, make_response
from openai import OpenAI
import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
from datetime import datetime
import json

# initialize logging - feel free to uncomment to view logs in console
# logging.basicConfig(level=logging.DEBUG) 
# logging.basicConfig(filename='app.log', level=logging.ERROR)

# flask backend
app = Flask(__name__)

# initialise openai
client = OpenAI(api_key="<openai key>")

# initialise gemini ai
genai.configure(api_key="<gemini ai key>")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# initialise BERT & configure Hugging Face API
tokenizer = BertTokenizer.from_pretrained("pace-group-51/fine-tuned-bert")
bert_model = BertForSequenceClassification.from_pretrained("pace-group-51/fine-tuned-bert")
HF_API_URL = "https://api-inference.huggingface.co/models/pace-group-51/fine-tuned-bert"
HF_HEADERS = {"Authorization": "Bearer <hugging face api key"}

@app.before_request
def log_cookies():
    logging.debug(f"Cookies: {request.cookies}")

# home route - handles GET requests and renders HTML template
@app.route('/')
def index():
    history = request.cookies.get('history')
    if history:
        history = json.loads(history)
    else:
        history = []
    return render_template('index.html', history=history)

# function to use BERT for text classification
def bert_predict(text):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    if response.status_code == 200:
        prediction = response.json()
        # print(prediction)
        if isinstance(prediction, list) and len(prediction) > 0:
            predicted_label = prediction[0][0]['label']
            return "This text is AI-generated." if predicted_label == "LABEL_1" else "This text is written by a human."
        else:
            return "Unexpected response format."
    else:
        logging.error(f"Hugging Face API Error: {response.status_code} - {response.text}")
        return "Error communicating with Hugging Face API."

# check route - handles POST requests /check URL
@app.route('/check', methods=['POST'])
def check():
    user_input = request.form.get('text', '')  # retrieves user text input
    logging.debug(f"User input: {user_input}") # logging

    # Get current history from cookies
    history = request.cookies.get('history')
    if history:
        history = json.loads(history)
    else:
        history = []

    if not user_input:  # if there's no input, render the existing history
        return render_template('index.html', openai_result="No text provided.", gemini_result="No text provided.", bert_result="No text provided.", text=user_input, history=history)

    # question variable
    question = "Is the following text written by AI? :\n\n" + user_input
    geminiQuestion = "Is the following text written by AI? Can you judge by rating either no, unlikely, uncertain, likely, yes? Can you quote what phrases appear to be AI or Human authored? Can you list reasons as to why it is AI or human authored? : \n\n" + user_input

    # openai response
    try:
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:ai-text-detect:A5jhKDL0",
            messages=[
                {"role": "system", "content": "You are an AI text detection tool."},
                {"role": "user", "content": question}
            ]
        )
        openai_result = completion.choices[0].message.content
    except Exception as e:  # error handling
        openai_result = f"Error: {str(e)}"
        logging.error(f"OpenAI Error: {str(e)}")  # Log error to app.log

    # gemini ai response
    try:
        gemini_response = gemini_model.generate_content(geminiQuestion)
        gemini_result = gemini_response.text
    except Exception as e:  # error handling
        gemini_result = f"Error: {str(e)}"
        logging.error(f"Gemini AI Error: {str(e)}")  # Log error to app.log

    # BERT response
    try:
        bert_result = bert_predict(user_input)
    except Exception as e:  # error handling
        bert_result = f"Error: {str(e)}"
        logging.error(f"BERT Error: {str(e)}")  # Log error to app.log

    # Update the history with trimmed results
    def limit_to_first_five_words(text):
        return ' '.join(text.split()[:5])

    # Prepare new entry
    new_entry = {
        'input': limit_to_first_five_words(user_input) + "...",
        'openai': openai_result,
        'gemini': limit_to_first_five_words(gemini_result) + "...",
        'bert': limit_to_first_five_words(bert_result),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Get current history from cookies
    history = request.cookies.get('history')
    if history:
        history = json.loads(history)
    else:
        history = []

    # Append new entry and limit to the last 10 entries
    history.append(new_entry)
    if len(history) > 10:
        history = history[-10:]  # Keep only the last 10 entries

    # Set updated history in cookies
    resp = make_response(render_template('index.html', openai_result=openai_result, gemini_result=gemini_result, bert_result=bert_result, text=user_input, history=history))
    resp.set_cookie('history', json.dumps(history), max_age=60*60*24)  # Cookie expires in 1 day

    logging.debug(f"Updated history: {history}")

    return resp
# run application
if __name__ == '__main__':
    app.run(debug=True)
