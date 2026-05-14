from ultralytics import YOLO
import cv2
import time
import threading
import math
import pyttsx3

# =========================
# YOLO MODEL
# =========================

YOLO_MODEL = "yolo11n.pt"

model = YOLO(YOLO_MODEL)

# =========================
# SETTINGS
# =========================

IDLE_THRESHOLD_SECONDS = 120
MOVEMENT_THRESHOLD = 20

# =========================
# WORKER STORAGE
# =========================

workers = {}

# =========================
# VOICE ENGINE
# =========================

engine = pyttsx3.init()

engine.setProperty('rate', 160)


def speak_warning(worker_id):

    text = f"Worker {worker_id}, please continue working"

    engine.say(text)
    engine.runAndWait()


def process_frame(frame, device):

    current_time = time.time()

    # Resize for speed
    frame = cv2.resize(frame, (960, 540))

    # =========================
    # YOLO TRACKING
    # =========================

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.4,
        iou=0.5,
        imgsz=640,
        half=True,
        device=device,
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()

        for box, tid in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # =========================
            # CENTER POINT
            # =========================

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            color = (0, 255, 0)
            status = "WORKING"

            # =========================
            # NEW WORKER
            # =========================

            if tid not in workers:

                workers[tid] = {
                    "last_position": (cx, cy),
                    "last_movement_time": current_time,
                    "alert_sent": False
                }

            else:

                prev_x, prev_y = workers[tid]["last_position"]

                # =========================
                # MOVEMENT DISTANCE
                # =========================

                distance = math.sqrt(
                    (cx - prev_x) ** 2 +
                    (cy - prev_y) ** 2
                )

                # =========================
                # WORKER MOVING
                # =========================

                if distance > MOVEMENT_THRESHOLD:

                    workers[tid]["last_movement_time"] = current_time
                    workers[tid]["alert_sent"] = False

                idle_time = (
                    current_time -
                    workers[tid]["last_movement_time"]
                )

                # =========================
                # IDLE DETECTION
                # =========================

                if idle_time > IDLE_THRESHOLD_SECONDS:

                    status = "IDLE"
                    color = (0, 0, 255)

                    # Send voice alert once
                    if not workers[tid]["alert_sent"]:

                        workers[tid]["alert_sent"] = True

                        threading.Thread(
                            target=speak_warning,
                            args=(tid,),
                            daemon=True
                        ).start()

                workers[tid]["last_position"] = (cx, cy)

            # =========================
            # DRAW BOUNDING BOX
            # =========================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # =========================
            # LABEL
            # =========================

            cv2.putText(
                frame,
                f"W-{tid} {status}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # =========================
            # CENTER POINT
            # =========================

            cv2.circle(
                frame,
                (cx, cy),
                5,
                color,
                -1
            )

    return frame