import logging
from flask import Flask, request, render_template, make_response, redirect, url_for, flash
from openai import OpenAI
import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
from datetime import datetime
import json
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import re

from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, EqualTo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# initialise logging - feel free to uncomment to view logs in console
# logging.basicConfig(level=logging.DEBUG) 
# logging.basicConfig(filename='app.log', level=logging.ERROR)

# flask backend
app = Flask(__name__)
app.secret_key = "secret"

# Configure SQL Alchemy
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///users.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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

# Find certain words in the response
def findWholeWord(w):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

# initialise rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# initialise openai
client = OpenAI(api_key="<openai API key>")

# initialise gemini ai
genai.configure(api_key="<gemini API key>")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# initialise BERT & configure Hugging Face API
tokenizer = BertTokenizer.from_pretrained("pace-group-51/fine-tuned-bert")
bert_model = BertForSequenceClassification.from_pretrained("pace-group-51/fine-tuned-bert")
HF_API_URL = "https://api-inference.huggingface.co/models/pace-group-51/fine-tuned-bert"
HF_HEADERS = {"Authorization": "Bearer <huggingface API key>"}

@app.before_request
def log_cookies():
    logging.debug(f"Cookies: {request.cookies}")

# sanitise user input
def sanitize_input(user_input):
    sanitized_input = re.sub(r'<.*?>', '', user_input)  # Remove any HTML tags
    return sanitized_input

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
    history = request.cookies.get('history')
    if history:
        history = json.loads(history)
    else:
        history = []
    return render_template('logs.html', history=history)

# function to use BERT for text classification
def bert_predict(text):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    if response.status_code == 200:
        prediction = response.json()
        # print(prediction)
        if isinstance(prediction, list) and len(prediction) > 0:
            predicted_label = prediction[0][0]['label']
            return "Yes, this text is AI-generated." if predicted_label == "LABEL_1" else "No, this text is written by a human."
        else:
            return "Unexpected response format."
    else:
        logging.error(f"Hugging Face API Error: {response.status_code} - {response.text}")
        return "Error communicating with Hugging Face API."

# check route - handles POST requests /check URL
@app.route('/check', methods=['POST'])
@login_required
@limiter.limit("5 per minute") # Limit to 5 requests per minute per IP
def check():
    user_input = request.form.get('text', '')  # Retrieve user text input
    user_input = sanitize_input(user_input)    # Sanitize user input
    logging.debug(f"Sanitized User input: {user_input}") # logging

    # Get current history from cookies
    history = request.cookies.get('history')
    if history:
        history = json.loads(history)
    else:
        history = []

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
        'openai': openai_result + "...",
        'bert': limit_to_first_five_words(bert_result) + "...",
        'gemini': limit_to_first_five_words(gemini_result) + "...",
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

    flash('A new log entry has been added.', 'info') # Log notification

    results_data = [0, 0]
    responses = [openai_result, gemini_result, bert_result]
    search_results = findWholeWord('yes')
    for response in responses:
        result = search_results(response)
        if result:
            results_data[0] += 1
        else:
            results_data[1] += 1

    def pie(data):
        plt.figure()  # Create a new figure
        plt.pie(data, labels=['Yes','No'], colors=['green', 'red'], autopct='%1.0f%%', pctdistance=0.85, explode=[0.05, 0.05])
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", transparent=True)
        plt.close()  # Close the figure to free memory
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
        return img_data

    def donut(data):
        plt.figure()  # Create a new figure
        ax = plt.subplot()
        wedges, text, autotext = ax.pie(data, colors=['green', 'red'], labels=['Yes', 'No'], autopct='%1.0f%%', pctdistance=0.85, explode=[0.05, 0.05])
        plt.setp(wedges, width=0.35)
        ax.set_aspect('equal')
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", transparent=True)
        plt.close()  # Close the figure to free memory
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
        return img_data

    # Set updated history in cookies
    resp = make_response(render_template('index.html', openai_result=openai_result, gemini_result=gemini_result, bert_result=bert_result, text=user_input, history=history, chart_data=donut(results_data)))
    resp.set_cookie('history', json.dumps(history), max_age=60*60*24)  # Cookie expires in 1 day

    logging.debug(f"Updated history: {history}")

    return resp

# run application
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
