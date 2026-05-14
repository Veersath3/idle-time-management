import subprocess
import cv2
import numpy as np

RTSP_URL = "rtsp://admin:!SoftDesigners1@192.168.0.124:554/Streaming/Channels/104"

# Native camera resolution
WIDTH = 1280
HEIGHT = 720

command = [
    r"C:\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe",

    # RTSP
    '-rtsp_transport', 'tcp',

    # Low latency
    '-fflags', 'nobuffer',
    '-flags', 'low_delay',

    # Faster startup
    '-analyzeduration', '0',
    '-probesize', '32',

    # Input
    '-i', RTSP_URL,

    # IMPORTANT:
    # Remove scaling for best clarity
    # DO NOT use scale filter

    # Output raw frames
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',

    '-an',
    '-sn',

    '-'
]

pipe = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=10**8
)

print("[INFO] High Clarity Stream Started")

frame_size = WIDTH * HEIGHT * 3

while True:

    raw_frame = pipe.stdout.read(frame_size)

    if len(raw_frame) != frame_size:
        continue

    frame = np.frombuffer(raw_frame, dtype=np.uint8)
    frame = frame.reshape((HEIGHT, WIDTH, 3))

    # Optional sharpening
    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    frame = cv2.filter2D(frame, -1, sharpen_kernel)

    cv2.imshow("High Clarity RTSP", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.kill()
cv2.destroyAllWindows()