from flask import Flask, request, jsonify
import transformers
import torch

# -------------------------
# Load Llama Model Once
# -------------------------
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    user_text = data["text"]
    #        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},

    messages = [
        {"role": "user", "content": user_text},
    ]

    try:
        output = pipeline(
            messages,
            max_new_tokens=256,
        )
        # Pipeline returns a list; take the last assistant segment
        result = output[0]["generated_text"][-1]["content"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": result})


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
