import requests

model_inputs = {"prompt": "Terrible service."}
res = requests.post("http://localhost:8000/", json=model_inputs)
print(res.json())
