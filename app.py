import logging
from flask import Flask, request, render_template
from openai import OpenAI
import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests

# initialize logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# flask backend
app = Flask(__name__)

# initialise openai
client = OpenAI(api_key="<OpenAI API Key>")

# initialise gemini ai
genai.configure(api_key="<Gemini API Key>")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# initialise BERT & configure Hugging Face API
tokenizer = BertTokenizer.from_pretrained("pace-group-51/fine-tuned-bert")
bert_model = BertForSequenceClassification.from_pretrained("pace-group-51/fine-tuned-bert")
HF_API_URL = "https://api-inference.huggingface.co/models/pace-group-51/fine-tuned-bert"
HF_HEADERS = {"Authorization": "Bearer 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"}  # API key after "Bearer "

# home route - handles GET requests and renders HTML template
@app.route('/')
def index():
    return render_template('index.html')

# function to use BERT for text classification
def bert_predict(text):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    if response.status_code == 200:
        prediction = response.json()
        # print(prediction)
        if isinstance(prediction, list) and len(prediction) > 0:
            predicted_label = prediction[0][0]['label']
            return "AI" if predicted_label == "LABEL_1" else "Human"
        else:
            return "Unexpected response format."
    else:
        logging.error(f"Hugging Face API Error: {response.status_code} - {response.text}")
        return "Error communicating with Hugging Face API."

# check route - handles POST requests /check URL
@app.route('/check', methods=['POST'])
def check():
    user_input = request.form.get('text', '')  # retrieves user text input
    if not user_input:  # error handling
        return render_template('index.html', openai_result="No text provided.", gemini_result="No text provided.", bert_result="No text provided.", text=user_input)

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

    return render_template('index.html', openai_result=openai_result, gemini_result=gemini_result, bert_result=bert_result, text=user_input)

# run application
if __name__ == '__main__':
    app.run(debug=True)
