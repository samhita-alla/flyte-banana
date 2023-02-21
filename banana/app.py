import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    device = 0 if torch.cuda.is_available() else -1
    model = AutoModelForSequenceClassification.from_pretrained(
        "model", num_labels=5
    ).to(device)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    device = 0 if torch.cuda.is_available() else -1

    # Parse out your arguments
    prompt = model_inputs.get("prompt", None)
    if prompt == None:
        return {"message": "No prompt provided"}

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased").to(device)
    encoding = tokenizer(prompt, return_tensors="pt")

    # Run the model
    outputs = model(**encoding)
    prediction = outputs.logits.argmax(-1)

    # Return the result as a dictionary
    return {"result": prediction.item()}
