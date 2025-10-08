from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to saved_model inside the same folder
model_path = os.path.join(os.path.dirname(__file__), "saved_model")

# Load model and tokenizer
loaded_model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example text to predict
text = "President of India is poisoned by the president of Bhutan"
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = loaded_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    labels = ["Fake", "Real"]
    print("Predicted label:", labels[predicted_class])

