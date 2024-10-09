# Standard Library Imports
import logging
import re
import math
import io
import base64
import csv
import os
from datetime import datetime
import json

# Flask and Related Imports
from flask import Flask, request, render_template, make_response, redirect, url_for, flash, send_file, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, EqualTo

# Security and Authentication
from werkzeug.security import generate_password_hash, check_password_hash

# OpenAI and Generative AI Imports
from openai import OpenAI
import google.generativeai as genai

# Transformers and Machine Learning Imports
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests

# Visualization Libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Environment Variables
from dotenv import load_dotenv

# initialise logging - feel free to uncomment to view logs in console
# logging.basicConfig(level=logging.DEBUG) 
# logging.basicConfig(filename='app.log', level=logging.ERROR)

# flask backend
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Load environment variables from .env file
load_dotenv()

# Configure SQL Alchemy
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///users.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Database User Model
class User(db.Model, UserMixin): 
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class RegistrationForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(
        min=4, max=20)], render_kw={'placeholder': "Username"})
    password = PasswordField(validators=[InputRequired(), Length(
        min=4, max=20)], render_kw={"placeholder": "Password"})
    password2 = PasswordField(validators=[InputRequired(), EqualTo('password', message='Passwords must match.')], render_kw={"placeholder": "Repeat Password"})
    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                "That username already exists. Please choose a different one."
            )

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(
        min=4, max=20)], render_kw={'placeholder': "Username"})
    password = PasswordField(validators=[InputRequired(), Length(
        min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")


# initialise rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# initialise openai
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# initialise gemini ai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# initialise BERT & configure Hugging Face API
tokenizer = BertTokenizer.from_pretrained("pace-group-51/fine-tuned-bert")
bert_model = BertForSequenceClassification.from_pretrained("pace-group-51/fine-tuned-bert")
HF_API_URL = "https://api-inference.huggingface.co/models/pace-group-51/fine-tuned-bert"
HF_HEADERS = {"Authorization": "Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

@app.before_request
def log_cookies():
    logging.debug(f"Cookies: {request.cookies}")

# sanitise user input
def sanitize_input(user_input):
    sanitized_input = re.sub(r'<.*?>', '', user_input)  # Remove any HTML tags
    return sanitized_input

# Get current History from cookies
def get_history():
    history = request.cookies.get('history')
    if history:
        history = json.loads(history)
    else:
        history = []
    return history

# home route - handles GET requests and renders HTML template
@app.route('/home')
@login_required
def home():
    return render_template('index.html')

# Login page
@app.route('/', methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Successfully Logged In.', 'info')
            return redirect(url_for('home'))
        elif not user:
            flash('User does not exist.', 'error')
        elif not user.check_password(form.password.data):
            flash('Wrong Password.', 'error')
    return render_template('login.html', form=form)
    

# Register
@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("3 per minute")
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        new_user = User(username=form.username.data)
        new_user.set_password(form.password.data)
        db.session.add(new_user)
        db.session.commit()
        flash(form.username.data + ' Successfully Registered', 'info')
        login_user(new_user)
        return redirect(url_for('home'))
    if form.errors:
        for field in form:
            for error in field.errors:
                flash(error, 'error')
    return render_template('register.html', form=form)


# Logout
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# Route to download history as a JSON file
@app.route('/download_history')
@login_required
@limiter.limit("1 per minute")
def download_history():
    
    if not get_history():
        flash('No logs found to download.', 'error')
        return redirect(url_for('logs'))
    
    file_path = 'logs.csv' # Define the file path for the CSV
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['input', 'openai', 'bert', 'gemini', 'timestamp'])
        writer.writeheader()
        for entry in get_history():
            writer.writerow(entry)

    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logging.error(f'Error downloading logs: {str(e)}')
        flash('Unable to download logs.')
        return redirect(url_for('logs'))

# about route
@app.route('/about/purpose')
@login_required
def purpose():
    return render_template('purpose.html')

@app.route('/about/ai-models')
@login_required
def ai_models():
    return render_template('ai_models.html')

# logs route
@app.route('/logs')
@login_required
def logs():
    return render_template('logs.html', history=get_history())

# function to use BERT for text classification
def bert_predict(text):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    if response.status_code == 200:
        prediction = response.json()
        if isinstance(prediction, list) and len(prediction) > 0:
            predicted_label = prediction[0][0]['label']
            if predicted_label == "LABEL_1":
                ai_probability = round(prediction[0][0]['score'] * 100, 2)
            else:
                ai_probability = round(100 - round(prediction[0][0]['score'] * 100), 2)
            result = f"Yes, this text is most likely AI-generated. (AI Probability: {ai_probability}%)" if predicted_label == "LABEL_1" else f"No, this text is most likely written by a Human. (AI Probability: {ai_probability}%)"
            return result, ai_probability
        else:
            return "Unexpected response format.", None

    else:
        logging.error(f"Hugging Face API Error: {response.status_code} - {response.text}")
        return "Error communicating with Hugging Face API.", None

gemini_probability_vals = {
    "most likely": 95,
    "highly unlikely": 5,
    "unlikely": 25,
    "likely": 75,
    "uncertain": 50
}

# check route - handles POST requests /check URL
@app.route('/check', methods=['POST'])
@login_required
@limiter.limit("5 per minute") # Limit to 5 requests per minute per IP
def check():
    user_input = request.form.get('text', '')  # Retrieve user text input
    user_input = sanitize_input(user_input)    # Sanitize user input
    logging.debug(f"Sanitized User input: {user_input}") # logging

    if not user_input:  # if there's no input, render the existing history
        return render_template('index.html', openai_result="No text provided.", gemini_result="No text provided.", bert_result="No text provided.", text=user_input, history=history)

    # question variables
    question = "Is the following text written by AI? :\n\n" + user_input
    geminiQuestion = "Is the following text AI-generated? Can you respond with a simple explanation, without making any headings? Include your rating (most likely, likely, uncertain, unlikely, highly unlikely), any phrases that appear AI-generated and/or human-authored, and your reasoning as to why you gave that rating: \n\n" + user_input

    # openai response
    try:
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:ai-text-detect:A5jhKDL0",
            messages=[
                {"role": "system", "content": "You are an AI text detection tool."},
                {"role": "user", "content": question}
            ],
            temperature=0,
            logprobs=True
        )
        predicted_label = completion.choices[0].message.content
        if "yes" in predicted_label.lower():
            openai_ai_probability = round(math.exp(completion.choices[0].logprobs.content[0].logprob) * 100, 2)
        else:
            openai_ai_probability = round(100 - round(math.exp(completion.choices[0].logprobs.content[0].logprob) * 100, 2),2)
        openai_result = f"{predicted_label}. (AI Probability: {openai_ai_probability}%)"
    except Exception as e:  # error handling
        openai_result = f"Error: {str(e)}"
        openai_ai_probability = None
        logging.error(f"OpenAI Error: {str(e)}")  # Log error to app.log

    # gemini ai response
    try:
        gemini_response = gemini_model.generate_content(geminiQuestion)
        gemini_result = gemini_response.text
        for phrase, value in gemini_probability_vals.items():
            if phrase in gemini_result.lower():
                gemini_ai_probability = value
                break
        gemini_result += f" (AI Probability: {gemini_ai_probability}%)"
    except Exception as e:  # error handling
        gemini_result = f"Error: {str(e)}"
        gemini_ai_probability = None
        logging.error(f"Gemini AI Error: {str(e)}")  # Log error to app.log

    # BERT response
    try:
        bert_result, bert_ai_probability = bert_predict(user_input)
    except Exception as e:  # error handling
        bert_result = f"Error: {str(e)}"
        logging.error(f"BERT Error: {str(e)}")  # Log error to app.log

    # calculate average AI probability of the models (excluding any models returning an error)
    valid_probabilities = [p for p in [openai_ai_probability, bert_ai_probability, gemini_ai_probability] if p is not None]
    avg_ai_probability = round(sum(valid_probabilities) / len(valid_probabilities), 2) if valid_probabilities else None
    avg_human_probability = 100 - avg_ai_probability if valid_probabilities else None

    # Update the history with trimmed results
    def limit_to_first_five_words(text):
        return ' '.join(text.split()[:5])

    # Prepare new entry
    new_entry = {
        'input': limit_to_first_five_words(user_input) + "...",
        'openai': openai_result + "...",
        'bert': limit_to_first_five_words(bert_result) + "...",
        'gemini': limit_to_first_five_words(gemini_result) + "...",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Get current history from cookie
    history = get_history()

    # Append new entry and limit to the last 10 entries
    history.append(new_entry)
    if len(history) > 10:
        history = history[-10:]  # Keep only the last 10 entries

    flash('A new log entry has been added.', 'info') # Log notification

    def donut(data):
        plt.figure()  # Create a new figure
        ax = plt.subplot()
        wedges, text, autotext = ax.pie(data, colors=['red', 'green'], labels=['AI', 'Human'], autopct='%1.0f%%', pctdistance=0.85, explode=[0.05, 0.05])
        plt.setp(wedges, width=0.35)
        ax.set_aspect('equal')
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", transparent=True)
        plt.close()  # Close the figure to free memory
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
        return img_data

    # Set updated history in cookies
    resp = make_response(render_template('index.html', openai_result=openai_result, gemini_result=gemini_result, bert_result=bert_result, text=user_input, history=history, chart_data=donut([avg_ai_probability, avg_human_probability])))
    resp.set_cookie('history', json.dumps(history), max_age=60*60*24)  # Cookie expires in 1 day

    logging.debug(f"Updated history: {history}")

    return resp

# run application
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
