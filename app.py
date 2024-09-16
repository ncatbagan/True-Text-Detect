import logging
from flask import Flask, request, render_template
from openai import OpenAI
import google.generativeai as genai

# initialize logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# flask backend
app = Flask(__name__)

# initialise openai
client = OpenAI(api_key="<insert openai apikey")

# initialise gemini ai
genai.configure(api_key="<insert gemini apikey")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# home route - handles GET requests and renders HTML template
@app.route('/')
def index():
    return render_template('index.html')

# check route - handes POST requests /check URL
@app.route('/check', methods=['POST'])
def check():
    user_input = request.form.get('text', '') # retrieves user text input
    if not user_input: # error handling
        return render_template('index.html', openai_result="No text provided.", gemini_result="No text provided.", text=user_input)

    # question variable
    question = "Is the following text written by AI? :\n\n" + user_input

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
    except Exception as e: # error handling
        openai_result = f"Error: {str(e)}"
        logging.error(f"OpenAI Error: {str(e)}")  # Log error to app.log

    # gemini ai response
    try:
        gemini_response = gemini_model.generate_content(question)
        gemini_result = gemini_response.text
    except Exception as e: # error handling
        gemini_result = f"Error: {str(e)}"
        logging.error(f"Gemini AI Error: {str(e)}")  # Log error to app.log

    return render_template('index.html', openai_result=openai_result, gemini_result=gemini_result, text=user_input)

# rn application
if __name__ == '__main__':
    app.run(debug=True)
