import streamlit as st
import gdown
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# URL for your model weights stored on Google Drive
MODEL_URL = 'https://drive.google.com/drive/folders/1bx4hZnDOxY42RtCLBUH9qrac10d3sQo5?usp=sharing '  # Replace with actual file ID from Googlhttps://drive.google.com/drive/folders/1bx4hZnDOxY42RtCLBUH9qrac10d3sQo5?usp=sharinge Drive

# Function to download the model weights
@st.cache_resource
def download_model():
    output = 'bert_email_subject_model.bin'
    gdown.download(MODEL_URL, output, quiet=False)
    return output

# Download the model weights
model_path = download_model()

# Load the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained(model_path)
model.eval()

# Streamlit app
st.title("Email Subject Prediction App")

email_body = st.text_area("Enter the email body text", "")

if st.button("Predict Subject"):
    if email_body.strip():
        # Tokenize the input
        inputs = tokenizer(email_body, return_tensors='pt', truncation=True, padding=True, max_length=100)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert logits to predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0], skip_special_tokens=True)
        predicted_text = tokenizer.convert_tokens_to_string(predicted_tokens)

        st.write(f"Predicted Subject: {predicted_text}")
    else:
        st.write("Please enter valid email body text.")
