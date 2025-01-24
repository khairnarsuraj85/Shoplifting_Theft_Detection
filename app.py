from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load your trained YOLO model
try:
    model = YOLO('./model/best.pt')  # Replace with your model's .pt file
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Check if the image is part of the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Retrieve the image from the request
        file = request.files['image']
        image_data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Validate the image
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Run the YOLO model on the image
        results = model.predict(img)

        # Parse results
        detections = []
        for result in results[0].boxes:
            detections.append({
                'label': model.names[int(result.cls)],  # Map class index to label
                'confidence': float(result.conf),      # Confidence score
                'bbox': result.xyxy.tolist()          # Bounding box coordinates
            })

        return jsonify({'detections': detections})

    except Exception as e:
        # Catch and return any error that occurs
        return jsonify({'error': f"An error occurred: {e}"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Shoplifting Theft Detection API is running!'})

if __name__ == '__main__':
    # Set debug=True for easier debugging during development
    app.run(host='0.0.0.0', port=5000, debug=True)
