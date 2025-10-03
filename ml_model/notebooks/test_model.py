model_path = r"C:\Users\Admin\Desktop\NewsWebsiteNLPembedded\ml_model\notebooks\saved_model"

from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch

device = torch.device("cpu")  # safe for now
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")
