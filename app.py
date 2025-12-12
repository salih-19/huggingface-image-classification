from flask import Flask, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Initialize TinyLlama pipeline
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    user_text = data["text"]

    # Wrap user input in chat format
    messages = [
        {"role": "user", "content": user_text}
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate text
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    generated_text = outputs[0]["generated_text"]

    return jsonify({"response": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
