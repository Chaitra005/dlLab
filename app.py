from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import nltk

# Download NLTK resources (if not already installed)
nltk.download('punkt')

app = Flask(__name__)

# Load the pre-trained model and tokenizer
MODEL_PATH = 'model/bert_email_subject_model'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Preprocess the input data
def preprocess_input(email_body, max_len=100):
    encoding = tokenizer(
        email_body,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoding

# Perform inference
def generate_subject_line(email_body):
    model.eval()
    encoding = preprocess_input(email_body)
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0], skip_special_tokens=True)
    predicted_subject = tokenizer.convert_tokens_to_string(predicted_tokens)
    
    return predicted_subject

# Define Flask route for generating the subject line
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_body = request.form['email_body']
        
        if not email_body:
            return jsonify({"error": "No email body provided!"})
        
        # Generate the subject line
        subject_line = generate_subject_line(email_body)
        return jsonify({"subject_line": subject_line})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
