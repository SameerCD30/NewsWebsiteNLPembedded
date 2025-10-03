import os
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch

# Check files
print(os.listdir("./saved_model"))

# Load model
device = torch.device("cpu")  # safe for now
model = DistilBertForSequenceClassification.from_pretrained("./saved_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("./saved_model")

print("Model and tokenizer loaded successfully!")
