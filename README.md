# MQ PACE S2 2024 - True Text Detect 
<h2>Project Details</h2>
<p>This project addresses the emerging challenges posed by advancements in AI-generated content. While AI technology holds significant potential for positive applications, it also poses risks by generating high-quality, deceptive content that is difficult to distinguish from human-produced material. Malicious actors can exploit these capabilities to create false, credible-sounding content and profiles at scale. This project focuses on solving the core challenge of accurately detecting machine-generated content. Specifically, developing robust techniques to identify AI-generated content, and ensuring the detection of erroneous, hallucinating, or misleading information.</p>

<h2>Features</h2>
<ul>
  <li>User authentication with registration and login functionality. </li> 
  <li>Integration with OpenAI and Gemini AI for text analysis. </li>  
  <li>Utilisation of a BERT model for text classification. </li> 
  <li>Rate limiting to prevent abuse of the service. </li> 
  <li>History tracking of user inputs with the ability to download logs. </li>  
  <li>Visual representation of AI vs. Human probability through pie charts. </li> 
</ul>

<h2>Installation</h2>
<p>To set up the application locally, follow these steps:

1. **Clone the repository:**
    ```bash
   git clone https://github.com/ncatbagan/True-Text-Detect
   ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables: Create a .env file in the root directory with the following variables:**
    ```bash
    SECRET_KEY=your_secret_key
    OPENAI_API_KEY=your_openai_api_key
    GEMINI_API_KEY=your_gemini_api_key
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    ```
</p>

<h2>Usage</h2>
<p>
  
1. **Run the application:** This demo is using VSCode to run our code.
<br> ![Demo1](https://github.com/user-attachments/assets/5e9a3205-a0ca-4ce5-a88d-c90b57cbc3af)

2. **Access the application:** Open your web browser and navigate to http://127.0.0.1:5000.

3. **Register and log in:** Follow the prompts to create a new account and log in.
<br> ![Demo2](https://github.com/user-attachments/assets/0e862cf4-0d40-47bd-9d6d-dc05156f2d21)

4. **Check text:** Use the input form to submit text for analysis. The application will provide feedback on whether the text is likely AI-generated.
<br> ![Demo3](https://github.com/user-attachments/assets/8f273810-3474-4013-8bc1-1fdf008d4058)

5. **Download history:** You can download your previous analyses as a CSV file for record-keeping.
<br> ![Demo4](https://github.com/user-attachments/assets/37e9f625-6428-45d4-a599-5fafe398829c)
</p>

<h2>License</h2>

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). 

You are free to:
<ul>
<li>Share — copy and redistribute the material in any medium or format </li>
<li>Adapt — remix, transform, and build upon the material </li>
</ul>

**Under the following terms:**
<ul>
<li> Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. </li>
</ul>
<p>For more details, please see the <a href="LICENSE">LICENSE</a> file or visit the <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Website</a>.</p>


<h2>Acknowledgements</h2>
<ul>
  <li><a href="https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text" target="_blank">Training/Fine-tuning dataset</a></li>
  <li><a href="https://flask.palletsprojects.com/" target="_blank">Flask</a> for the web framework.</li>
  <li><a href="https://openai.com/" target="_blank">OpenAI</a> and <a href="https://google.com/generativeai" target="_blank">Gemini AI</a> for the AI models.</li>
  <li><a href="https://huggingface.co/transformers/" target="_blank">Hugging Face Transformers</a> for BERT.</li>
  <li><a href="https://matplotlib.org/" target="_blank">Matplotlib</a> for visualisation.</li>

</ul>

<h2>Contributors</h2>

**Zarif Ahmed Anik** - (https://github.com/Anik9076)<br>
**Norreigne Catbagan** - (https://github.com/ncatbagan)<br>
**Simon Ding** <br>
**Sawyer James Rush** - (https://github.com/SawyerRush)<br>
**Mark Kneale** - (https://github.com/mark-kneale-2)<br>
**Chelsi Patel** - (https://github.com/Chelsip)<br>
