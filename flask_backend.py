from flask import Flask, Response
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from Streamlit or other frontends

# Try to open the camera
camera = cv2.VideoCapture(3)  # Use appropriate index for your camera

if not camera.isOpened():
    print("Error: Could not access camera.")
else:
    print("Camera successfully accessed!")

def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/preview')
def preview():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    ret, frame = camera.read()
    if not ret:
        return "Error: Failed to capture frame from camera.", 500

    # Encode frame as JPEG
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return "Error: Failed to encode frame.", 500

    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/')
def home():
    return "Flask Camera API running. Use /preview for streaming or /capture to capture a single frame."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)










# # flask_backend.py
# from flask import Flask, send_file, Response
# from flask_cors import CORS
# import cv2
# from io import BytesIO
# import requests
# import numpy as np

# app = Flask(__name__)
# CORS(app)

# # DroidCam (USB): device index, e.g., 3 or 0
# droidcam = cv2.VideoCapture(3)

# # IP Webcam (URL stream)
# IP_WEBCAM_URL = "http://192.168.29.79:8001/video"

# @app.route('/preview_droid')
# def preview_droid():
#     ret, frame = droidcam.read()
#     if not ret:
#         return "DroidCam error", 500
#     _, jpeg = cv2.imencode('.jpg', frame)
#     return send_file(BytesIO(jpeg.tobytes()), mimetype='image/jpeg')

# @app.route('/preview_ipwebcam')
# def preview_ipwebcam():
#     try:
#         response = requests.get(IP_WEBCAM_URL, stream=True, timeout=3)
#         if response.status_code == 200:
#             return Response(response.raw, content_type='multipart/x-mixed-replace; boundary=frame')
#         else:
#             return "IP Webcam feed error", 500
#     except:
#         return "IP Webcam connection failed", 500

# @app.route('/capture_droid')
# def capture_droid():
#     ret, frame = droidcam.read()
#     if not ret:
#         return "Capture error", 500
#     _, jpeg = cv2.imencode('.jpg', frame)
#     return send_file(BytesIO(jpeg.tobytes()), mimetype='image/jpeg')

# @app.route('/capture_ipwebcam')
# def capture_ipwebcam():
#     try:
#         img_resp = requests.get(IP_WEBCAM_URL.replace('/video', '/shot.jpg'))
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         frame = cv2.imdecode(img_arr, -1)
#         _, jpeg = cv2.imencode('.jpg', frame)
#         return send_file(BytesIO(jpeg.tobytes()), mimetype='image/jpeg')
#     except:
#         return "IP Webcam capture failed", 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
