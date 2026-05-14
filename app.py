from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import cv2
import torch

from camera import RTSPCamera
from tracker import process_frame

# =========================
# FLASK
# =========================

app = Flask(__name__)

socketio = SocketIO(
    app,
    cors_allowed_origins="*"
)

# =========================
# RTSP URL
# =========================

RTSP_URL = "rtsp://admin:!SoftDesigners1@192.168.0.124:554/Streaming/Channels/104"

# =========================
# DEVICE
# =========================

device = 0 if torch.cuda.is_available() else "cpu"

print(f"[INFO] Using Device: {device}")

# =========================
# CAMERA
# =========================

camera = RTSPCamera(RTSP_URL)


@app.route("/")
def index():

    return render_template("index.html")


# =========================
# TRACKING LOOP
# =========================

def tracking_loop():

    print("[INFO] Ultra Low Latency Tracking Started")

    while True:

        ret, frame = camera.read()

        if not ret:
            continue

        frame = process_frame(frame, device)

        cv2.imshow(
            "Construction Worker Monitoring",
            frame
        )

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    camera.release()

    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    tracking_thread = threading.Thread(
        target=tracking_loop,
        daemon=True
    )

    tracking_thread.start()

    print("[INFO] Dashboard → http://localhost:5001")

    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=False
    )