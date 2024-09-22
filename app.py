import logging
from flask import Flask, request, render_template
from openai import OpenAI
import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# initialize logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# flask backend
app = Flask(__name__)

# initialise openai
client = OpenAI(api_key="<OpenAI API Key>")

# initialise gemini ai
genai.configure(api_key="<Gemini API Key>")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert/')

# home route - handles GET requests and renders HTML template
@app.route('/')
def index():
    return render_template('index.html')

# function to use BERT for text classification
def bert_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "AI" if predicted_class == 1 else "Human"

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
