# from flask import Flask, request, jsonify
# import torch
# from transformers import pipeline

# app = Flask(__name__)

# # Initialize TinyLlama pipeline
# pipe = pipeline(
#     "text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )

# @app.route("/generate", methods=["POST"])
# def generate():
#     data = request.get_json()
#     if not data or "text" not in data:
#         return jsonify({"error": "Missing 'text' field"}), 400

#     user_text = data["text"]

#     # Wrap user input in chat format
#     messages = [
#         {"role": "user", "content": user_text}
#     ]
#     prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     # Generate text
#     outputs = pipe(
#         prompt,
#         max_new_tokens=256,
#         do_sample=True,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.95
#     )

#     generated_text = outputs[0]["generated_text"]

#     return jsonify({"response": generated_text})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


from flask import Flask, request
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import base64
from io import BytesIO
import time

app = Flask(__name__)

# Path to the locally saved model
MODEL_PATH = "./model"

# HTML templates
INDEX_HTML = """
<!DOCTYPE html>
<html>
<body>
<form action="/predict" method="post" enctype="multipart/form-data">
    <label>Select image to upload:</label>
    <input type="file" name="image">
    <input type="submit" value="Upload Image">
</form>
</body>
</html>
"""

RESPONSE_HTML = """
<!DOCTYPE html>
<html>
<body>
<img src="data:image/png;base64,{img_base64}" alt="image" width="500" height="600">
<form action="/predict" method="post" enctype="multipart/form-data">
    <label>Select image to upload:</label>
    <input type="file" name="image">
    <input type="submit" value="Upload Image">
</form>
<p>Prediction: {prediction}</p>
<p>Score: {score:.2f}</p>
<p>Time taken: {time_taken:.2f} seconds</p>
</body>
</html>
"""

# Load the model and processor from the local path
model = ResNetForImageClassification.from_pretrained(MODEL_PATH, local_files_only=True)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)


@app.route("/")
def index():
    """Render the upload page."""
    return INDEX_HTML


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, make predictions, and return results."""
    start_time = time.time()

    # Get the uploaded image file
    img_file = request.files.get("image")
    if not img_file:
        return "No image uploaded. Please upload a valid image."

    # Process the image for prediction
    img = Image.open(img_file)
    inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        confidence_score = torch.softmax(logits, dim=-1)[0, predicted_label].item()

    # Encode the image in Base64 to display it in the HTML response
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Calculate elapsed time
    time_taken = time.time() - start_time

    # Return the HTML with prediction results
    return RESPONSE_HTML.format(
        img_base64=img_base64,
        prediction=predicted_label,
        score=confidence_score,
        time_taken=time_taken,
    )


# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
