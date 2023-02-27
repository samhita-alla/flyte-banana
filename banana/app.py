import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    with open("model_metadata.json") as f:
        model_metadata = json.load(f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_metadata["repo"],
        num_labels=5,
        revision=model_metadata["sha"],
    ).to(device)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get("prompt", None)
    if prompt == None:
        return {"message": "No prompt provided"}

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoding = tokenizer(
        prompt, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)

    # Run the model
    outputs = model(**encoding)
    prediction = outputs.logits.argmax(-1)

    # Return the result as a dictionary
    return {"result": prediction.item()}
