import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import nltk

# Download NLTK resources (if not already installed)
nltk.download('punkt')

# Load the pre-trained model and tokenizer
MODEL_PATH = 'model/bert_email_subject_model'  # Update this path to your model
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

# Streamlit interface
st.title("Email Subject Prediction App")

email_body = st.text_area("Enter the email body text")

if st.button("Predict Subject"):
    if email_body.strip():
        subject_line = generate_subject_line(email_body)
        st.write(f"Predicted Subject: {subject_line}")
    else:
        st.write("Please enter valid email body text.")
