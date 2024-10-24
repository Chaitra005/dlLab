import streamlit as st
import gdown
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig

# Google Drive file IDs for the model weights and config file
WEIGHTS_URL = 'https://drive.google.com/file/d/1Ak-TYo4lB31tjfRe_dkiLf55VbdvC48m/view?usp=drive_link'
CONFIG_URL = 'https://drive.google.com/file/d/1S7TZKENa2hMMMoK6oxerCUC6vEijjDBI/view?usp=drive_link'

# Function to download files from Google Drive
@st.cache_resource
def download_files():
    weights_output = 'pytorch_model.bin'
    config_output = 'config.json'

    # Download the model weights and config file
    gdown.download(WEIGHTS_URL, weights_output, quiet=False)
    gdown.download(CONFIG_URL, config_output, quiet=False)
    
    return weights_output, config_output

# Download the model weights and config
weights_path, config_path = download_files()

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Load the config and model
config = BertConfig.from_json_file(config_path)
model = BertForTokenClassification(config)

# Load the weights into the model
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
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
