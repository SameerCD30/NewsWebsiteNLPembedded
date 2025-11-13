from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import torch

app = FastAPI()

MODEL_DIR = "best_bert_issue_detector"

# Load tokenizer + model (safetensors is detected automatically)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

class Input(BaseModel):
    text: str

@app.post("/predict")
def predict(data: Input):
    inputs = tokenizer(
        data.text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()

    return {
    "is_issue": bool(predicted == 0),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
