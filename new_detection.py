import os
from ultralytics import YOLO
import cv2
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import pandas as pd
import base64
import ibm_boto3
from ibm_botocore.client import Config

cos_credentials = {
    "apikey": "tsn6rgA_EgyOxgcSrdzTVzq97cpRtHfRKs_1b8yn4Fqt",
    "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/741dad3858d54945bacefcd15f300ca2:2c95220e-f6d9-429c-90cf-b9bdad1a04ff:bucket:finalshoplifting",
    "endpoint": "https://s3.jp-tok.cloud-object-storage.appdomain.cloud",
    "bucket_name": "finalshoplifting"
}

def detect_shoplifting(video_path):
    #model_yolo = YOLO(r"C:/Users/khair/Downloads/shoplifting_Web/backend/model/best.pt")
    #model = xgb.Booster()
    #model.load_model(r"C:/Users/khair/Downloads/shoplifting_Web/backend/model/model_weights.json")
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
        model= xgb.Booster()
        model.load_model("model_weights.json")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    cap = cv2.VideoCapture(video_path)

    print('Total Frame', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    
    # Generate a unique output path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = fr"C:/Users/khair/Downloads/shoplifting_Web/backend/model/output video/{video_name}_output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_tot = 0

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model_yolo(frame, verbose=False)

            # Visualize the results on the frame
            annotated_frame = results[0].plot(boxes=False)

            for r in results:
                bound_box = r.boxes.xyxy
                conf = r.boxes.conf.tolist()
                keypoints = r.keypoints.xyn.tolist()

                print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

                for index, box in enumerate(bound_box):
                    if conf[index] > 0.75:
                        x1, y1, x2, y2 = box.tolist()
                        data = {}

                        # Initialize the x and y lists for each possible key
                        for j in range(len(keypoints[index])):
                            data[f'x{j}'] = keypoints[index][j][0]
                            data[f'y{j}'] = keypoints[index][j][1]

                        # print(f'Bounding Box {index}: {data}')
                        df = pd.DataFrame(data, index=[0])
                        dmatrix = xgb.DMatrix(df)
                        cut = model.predict(dmatrix)
                        binary_predictions = (cut > 0.5).astype(int)
                        print(f'Prediction: {binary_predictions}')

                        if binary_predictions == 0:
                            conf_text = f'Suspicious ({conf[index]:.2f})'
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 7, 58), 2)
                            cv2.putText(annotated_frame, conf_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 7, 58), 2)
                        if binary_predictions == 1:
                            conf_text = f'Normal ({conf[index]:.2f})'
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (57, 255, 20), 2)
                            cv2.putText(annotated_frame, conf_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (57, 255, 20), 2)


            cv2.imshow('Frame', annotated_frame)

            out.write(annotated_frame)
            frame_tot += 1
            print('Processed Frame:', frame_tot)

            if cv2.waitKey(1) & 0xFF == ord("q"):
               break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage:
video_path = r"C:/Users/khair/Downloads/shoplifting_Web/backend/model/Shoplifting.mp4"
detect_shoplifting(video_path)

