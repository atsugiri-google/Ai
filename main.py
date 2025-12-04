from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


app = Flask(__name__)


model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


@app.route('/api/chat', methods=['POST'])
def chat():
data = request.get_json()
user_msg = data.get("message", "")


inputs = tokenizer(user_msg, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=60)
reply = tokenizer.decode(outputs[0], skip_special_tokens=True)


return jsonify({"reply": reply})


if __name__ == '__main__':
app.run(host='0.0.0.0', port=8080)
