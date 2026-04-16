import requests, os
from dotenv import load_dotenv
load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_TOKEN   = os.getenv("HF_TOKEN")

headlines = [
    "Apple stock rises on strong earnings",
    "Tesla recalls 500000 vehicles over safety concerns",
    "Gold holds steady amid market uncertainty"
]

response = requests.post(
    HF_API_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    json={"inputs": headlines},
    timeout=30
)

import json
print(json.dumps(response.json(), indent=2))