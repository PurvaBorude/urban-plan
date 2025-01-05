import streamlit as st
from transformers import AutoTokenizer, AutoModelForPreTraining
import torch

# Function to load the model and tokenizer
@st.cache_resource
def load_model():
    model_path = "./InLegalBERT"  # Local directory of the cloned model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForPreTraining.from_pretrained(model_path)
    
    # Confirm that the model and tokenizer are loaded
    st.write("Model and Tokenizer loaded successfully!")
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model()

# Streamlit user input
st.title("Legal Judgment Predictor")
case_details = st.text_area("Enter case details here:")

# Test the model with a static input (for debugging)
test_input = "This is a test case example."
if st.button("Test Model with Static Input"):
    inputs = tokenizer(test_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    st.write("Test case logits:", outputs.logits)

# Prediction when button is pressed
if st.button("Predict"):
    if case_details.strip():
        # Tokenize input from user
        inputs = tokenizer(case_details, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Run model inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits and display them
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            st.write("Logits (Raw Output):", logits)
        else:
            st.write("Model output does not have 'logits' attribute.")
        
        # Example: Get the predicted class (if it's a classification task)
        prediction = logits.argmax(dim=-1)
        st.write(f"Predicted Class: {prediction.item()}")
        
        st.success("Prediction complete!")
    else:
        st.warning("Please enter valid case details.")
