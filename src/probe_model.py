"""Quick probe: show raw response from a model to diagnose parsing issues."""
import os
from dotenv import load_dotenv
from ollama import Client

load_dotenv()
key = os.getenv("OLLAMA_KEY")
client = Client(host="https://ollama.com", headers={"Authorization": f"Bearer {key}"})

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 6},
        "rationale": {"type": "string"},
    },
    "required": ["score"],
}

resp = client.chat(
    model="glm-5.2:cloud",
    messages=[
        {"role": "system", "content": "You are an essay rater. Return JSON with integer 'score' 1-6 and 'rationale'."},
        {"role": "user", "content": "Score this short essay: 'The dog ran fast.' Assign score 1-6."},
    ],
    format=SCHEMA,
    options={"temperature": 0},
)
print(repr(resp["message"]["content"]))
