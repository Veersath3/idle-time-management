import os
import cv2
import time
import torch
import argparse
import threading

from queue import Queue
from datetime import datetime
from ultralytics import YOLO

from flask import Flask, jsonify, render_template_string
from flask_socketio import SocketIO

# =========================================================
# ULTRA LOW LATENCY RTSP SETTINGS
# =========================================================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0|"
    "probesize;32|"
    "analyzeduration;0"
)

# =========================================================
# CONFIG
# =========================================================

YOLO_MODEL = "yolo11n.pt"     # FASTEST
WEB_PORT = 5001

CONFIDENCE = 0.4
TRACKER_IOU = 0.5

IDLE_THRESHOLD_SEC = 120
MOVE_THRESHOLD_PX = 20

FRAME_SKIP = 1

workers = {}
alarm_log = []

# =========================================================
# GPU
# =========================================================

if torch.cuda.is_available():
    DEVICE = "cuda:0"
    GPU_NAME = torch.cuda.get_device_name(0)

    print(f"[GPU] Using: {GPU_NAME}")

    torch.backends.cudnn.benchmark = True

else:
    DEVICE = "cpu"
    GPU_NAME = "CPU"

print(f"[INFO] Device: {DEVICE}")

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# =========================================================
# HTML
# =========================================================

HTML = """
<!DOCTYPE html>
<html>

<head>

<title>Worker Tracker</title>

<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>

<style>

body{
    background:#0d1117;
    color:white;
    font-family:Arial;
    padding:20px;
}

.card{
    background:#161b22;
    padding:15px;
    margin-bottom:10px;
    border-radius:10px;
}

.active{
    border-left:5px solid #00ff00;
}

.idle{
    border-left:5px solid orange;
}

.alarm{
    border-left:5px solid red;
}

</style>

</head>

<body>

<h1>Construction Worker Tracker</h1>

<h3 id="fps">FPS: --</h3>

<div id="workers"></div>

<script>

const socket = io();

let speaking = {};

function speak(text){

    const u = new SpeechSynthesisUtterance(text);

    u.rate = 1;

    speechSynthesis.speak(u);
}

socket.on("state_update", function(data){

    const div = document.getElementById("workers");

    div.innerHTML = "";

    data.workers.forEach(w => {

        const d = document.createElement("div");

        d.className = "card " + w.status;

        d.innerHTML = `
            <h2>${w.id}</h2>
            <p>Status: ${w.status}</p>
            <p>Idle: ${w.idle_secs}s</p>
            <p>Zone: ${w.zone}</p>
        `;

        div.appendChild(d);

    });

});

socket.on("perf", function(data){

    document.getElementById("fps").innerHTML =
        "FPS: " + data.fps +
        " | Inference: " + data.inference_ms + "ms" +
        " | Device: " + data.device;

});

socket.on("alarm", function(data){

    const wid = data.worker_id;

    if(!speaking[wid]){

        speak("Please work " + wid);

        speaking[wid] = setInterval(() => {

            speak("Please work " + wid);

        }, 5000);
    }

});

socket.on("alarm_cleared", function(data){

    const wid = data.worker_id;

    if(speaking[wid]){

        clearInterval(speaking[wid]);

        delete speaking[wid];
    }

});

</script>

</body>
</html>
"""

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/workers")
def api_workers():
    return jsonify([w.to_dict() for w in workers.values()])

# =========================================================
# WORKER CLASS
# =========================================================

class WorkerState:

    def __init__(self, track_id, x, y):

        self.track_id = track_id
        self.worker_id = f"W-{track_id:03d}"

        self.x = x
        self.y = y

        self.status = "active"

        self.idle_since = None
        self.idle_secs = 0

        self.alarm_fired = False

        self.last_seen = time.time()

        self.zone = "Zone A"

    def update_position(self, x, y):

        dist = ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5

        self.last_seen = time.time()

        if dist > MOVE_THRESHOLD_PX:

            self.x = x
            self.y = y

            self.status = "active"

            self.idle_since = None
            self.idle_secs = 0

            self.alarm_fired = False

        else:

            if self.idle_since is None:
                self.idle_since = time.time()

            self.idle_secs = int(time.time() - self.idle_since)

            if self.idle_secs >= IDLE_THRESHOLD_SEC:
                self.status = "alarm"

            elif self.idle_secs > 10:
                self.status = "idle"

            else:
                self.status = "active"

    def to_dict(self):

        return {
            "id": self.worker_id,
            "status": self.status,
            "idle_secs": self.idle_secs,
            "zone": self.zone
        }

# =========================================================
# ULTRA LOW LATENCY CAMERA
# =========================================================

class VideoCaptureAsync:

    def __init__(self, src):

        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.q = Queue(maxsize=1)

        self.running = True

        t = threading.Thread(target=self.reader)

        t.daemon = True

        t.start()

    def reader(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                continue

            if not self.q.empty():

                try:
                    self.q.get_nowait()
                except:
                    pass

            self.q.put(frame)

    def read(self):

        return self.q.get()

    def release(self):

        self.running = False

        self.cap.release()

# =========================================================
# SOCKET PUSH
# =========================================================

def push_state():

    socketio.emit(
        "state_update",
        {
            "workers": [w.to_dict() for w in workers.values()]
        }
    )

# =========================================================
# TRACKING LOOP
# =========================================================

def tracking_loop(source):

    print(f"[INFO] Loading model: {YOLO_MODEL}")

    model = YOLO(YOLO_MODEL)

    model.to(DEVICE)

    use_half = DEVICE.startswith("cuda")

    cap = VideoCaptureAsync(source)

    print("[INFO] Ultra Low Latency Tracking Started")

    print(f"[INFO] Dashboard → http://localhost:{WEB_PORT}")

    fps_counter = 0
    fps = 0

    fps_timer = time.time()

    while True:

        frame = cap.read()

        if frame is None:
            continue

        fps_counter += 1

        start = time.time()

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            iou=TRACKER_IOU,
            device=DEVICE,
            half=use_half,
            imgsz=640,
            stream=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        inference_ms = int((time.time() - start) * 1000)

        if time.time() - fps_timer >= 1:

            fps = fps_counter

            fps_counter = 0

            fps_timer = time.time()

            socketio.emit(
                "perf",
                {
                    "fps": fps,
                    "inference_ms": inference_ms,
                    "device": DEVICE
                }
            )

        active_ids = set()

        for result in results:

            if result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()

            ids = result.boxes.id.int().cpu().numpy()

            for box, tid in zip(boxes, ids):

                x1, y1, x2, y2 = box

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                tid = int(tid)

                active_ids.add(tid)

                if tid not in workers:

                    workers[tid] = WorkerState(tid, cx, cy)

                    print(f"[NEW] {workers[tid].worker_id}")

                else:

                    prev = workers[tid].status

                    workers[tid].update_position(cx, cy)

                    now = workers[tid].status

                    if now == "alarm" and not workers[tid].alarm_fired:

                        workers[tid].alarm_fired = True

                        socketio.emit(
                            "alarm",
                            {
                                "worker_id": workers[tid].worker_id
                            }
                        )

                        print(f"[ALARM] {workers[tid].worker_id}")

                    if prev == "alarm" and now == "active":

                        socketio.emit(
                            "alarm_cleared",
                            {
                                "worker_id": workers[tid].worker_id
                            }
                        )

                # ====================================
                # DRAW
                # ====================================

                color = (
                    (0,255,0)
                    if workers[tid].status == "active"
                    else (0,165,255)
                    if workers[tid].status == "idle"
                    else (0,0,255)
                )

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2
                )

                label = (
                    f"{workers[tid].worker_id} "
                    f"{workers[tid].status} "
                    f"{workers[tid].idle_secs}s"
                )

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # ====================================
        # REMOVE OLD WORKERS
        # ====================================

        stale = []

        for tid, w in workers.items():

            if time.time() - w.last_seen > 5:
                stale.append(tid)

        for tid in stale:
            del workers[tid]

        push_state()

        cv2.imshow("Worker Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="rtsp://admin:!SoftDesigners1@192.168.0.124:554/Streaming/Channels/104"
    )

    args = parser.parse_args()

    source = args.source

    if source.isdigit():
        source = int(source)

    tracker_thread = threading.Thread(
        target=tracking_loop,
        args=(source,),
        daemon=True
    )

    tracker_thread.start()

    print(f"[INFO] Dashboard → http://localhost:{WEB_PORT}")

    socketio.run(
        app,
        host="0.0.0.0",
        port=WEB_PORT,
        debug=False
    )