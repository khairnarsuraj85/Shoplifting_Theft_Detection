from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import xgboost as xgb
import numpy as np
import pandas as pd
import base64
import os
import ibm_boto3
from ibm_botocore.client import Config

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# IBM COS credentials
cos_credentials = {
    "apikey": "tsn6rgA_EgyOxgcSrdzTVzq97cpRtHfRKs_1b8yn4Fqt",
    "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/741dad3858d54945bacefcd15f300ca2:2c95220e-f6d9-429c-90cf-b9bdad1a04ff:bucket:finalshoplifting",
    "endpoint": "https://s3.jp-tok.cloud-object-storage.appdomain.cloud",
    "bucket_name": "finalshoplifting"
}

# Create COS client
cos_client = ibm_boto3.client(
    "s3",
    ibm_api_key_id=cos_credentials["apikey"],
    ibm_service_instance_id=cos_credentials["resource_instance_id"],
    config=Config(signature_version="oauth"),
    endpoint_url=cos_credentials["endpoint"]
)

def check_and_download_model(local_model_path, cos_key):
    if not os.path.exists(local_model_path):
        try:
            cos_client.download_file(
                Bucket=cos_credentials["bucket_name"],
                Key=cos_key,
                Filename=local_model_path
            )
            print(f"Downloaded {cos_key} to {local_model_path}")
        except Exception as e:
            print(f"Error downloading {cos_key}: {e}")
            raise

def download_models():
    check_and_download_model("best.pt", "best.pt")
    check_and_download_model("model_weights.json", "model_weights.json")

try:
    download_models()

    model_yolo = YOLO("best.pt")
    model_xgb = xgb.Booster()
    model_xgb.load_model("model_weights.json")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

def process_frame(frame):
    try:
        results = model_yolo(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        detections = []
        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist() if hasattr(r.keypoints, "xyn") else []

            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    data = {}

                    if keypoints:
                        for j in range(len(keypoints[index])):
                            data[f'x{j}'] = keypoints[index][j][0]
                            data[f'y{j}'] = keypoints[index][j][1]

                    if data:
                        df = pd.DataFrame(data, index=[0])
                        dmatrix = xgb.DMatrix(df)
                        prediction = model_xgb.predict(dmatrix)
                        binary_prediction = int((prediction > 0.5).astype(int))
                    else:
                        binary_prediction = -1

                    label = "Suspicious" if binary_prediction == 0 else "Normal" if binary_prediction == 1 else "Unknown"
                    color = (0, 0, 255) if binary_prediction == 0 else (0, 255, 0) if binary_prediction == 1 else (0, 255, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f'{label} ({conf[index]:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    detections.append({
                        "label": label,
                        "confidence": conf[index],
                        "bounding_box": [x1, y1, x2, y2]
                    })

        return annotated_frame, detections
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, []

@app.route('/detect', methods=['POST'])
def detect_shoplifting_live():
    try:
        data = request.get_json()

        frame_data = data.get('frame')
        if not frame_data:
            return jsonify({"error": "No frame provided"}), 400

        img_data = base64.b64decode(frame_data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        annotated_frame, detections = process_frame(frame)
        return jsonify({"detections": detections})

    except Exception as e:
        print(f"Error in detect_shoplifting_live: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
