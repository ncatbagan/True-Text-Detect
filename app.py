from flask import Flask, request, render_template
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key="<insert GPT api key>")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    user_input = request.form.get('text', '')
    if not user_input:
        return render_template('index.html', result="No text provided.", text=user_input)

    question = "Is the following text written by AI? :\n\n" + user_input

    try:
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:ai-text-detect:A5jhKDL0",
            messages=[
                {"role": "system", "content": "You are an AI text detection tool."},
                {"role": "user", "content": question}
            ]
        )
        result = completion.choices[0].message.content
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', result=result, text=user_input)

if __name__ == '__main__':
    app.run(debug=True)