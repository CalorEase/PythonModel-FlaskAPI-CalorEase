import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
from collections import Counter
from google.cloud import storage
import tempfile

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])

interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
labels = ["ayam goreng", "nasi putih", "sambal", "tahu goreng", "tempe goreng"]

bucket_name = 'imagestoragedatabase'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './calorease-c3cd0-2778d4b3c4bd.json'
client = storage.Client()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200

@app.route("/upload", methods=['POST'])
def upload():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "No image sent"
                },
                "data": None
            }), 400

        image = request.files['image']
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(filename)
            blob.upload_from_file(image)

            # Create GCS path
            image_gcs_path = f"gs://{bucket_name}/{filename}"

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Image uploaded to GCS",
                },
                "data": {
                    "image_path": image_gcs_path
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Bad request or unsupported file type"
                },
                "data": None,
            }), 400

@app.route("/prediction", methods=['POST'])
def prediction():
    if request.method == "POST":
        data = request.json
        image_gcs_path = data.get('image_path', None)

        if image_gcs_path:
            # Read image from GCS
            blob = client.bucket(bucket_name).blob(image_gcs_path.split('/')[-1])
            temp_dir = tempfile.mkdtemp()
            temp_image = os.path.join(temp_dir, image_gcs_path.split('/')[-1])
            blob.download_to_filename(temp_image)

                # Preprocess image for TensorFlow Lite
            img = Image.open(temp_image).convert("RGB")
            img = img.resize((width, height))
            img = np.expand_dims(img, axis=0)
            if float_input:
                img = ((np.float32(img) - input_mean) / input_std)

            # Perform prediction with TensorFlow Lite model
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            classes = interpreter.get_tensor(output_details[3]['index'])[0]
            scores = interpreter.get_tensor(output_details[0]['index'])[0]

            # Retrieve predictions
            detections = []
            for i in range(len(scores)):
                if((scores[i] > 0.5) and (scores[i] <= 1.0)):
                    object_name = labels[int(classes[i])]
                    detections.append(object_name)

            detections_dict = dict(Counter(detections))
            
            formatted_predictions = []
            idx = 1
            for name, count in detections_dict.items():
                formatted_prediction = {
            "id": str(idx),
            "nama": name,
            "jumlah": str(count)
            }
            formatted_predictions.append(formatted_prediction)
            idx += 1

            # Clean up temporary directory and its contents
            if os.path.exists(temp_dir):
                os.remove(temp_image)
                os.rmdir(temp_dir)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Prediction successful",
                },
                "data": {
                    "prediction": formatted_predictions
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Bad request or missing image path"
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405


if __name__ == "__main__":
    app.run()
