import subprocess
import threading
import numpy as np
import cv2
import time


class RTSPCamera:

    def __init__(self, rtsp_url):

        self.WIDTH = 1280
        self.HEIGHT = 720

        self.frame_size = self.WIDTH * self.HEIGHT * 3

        self.latest_frame = None
        self.running = True

        self.command = [

            r"C:\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe",

            "-rtsp_transport", "tcp",

            "-fflags", "nobuffer",
            "-flags", "low_delay",

            "-analyzeduration", "0",
            "-probesize", "32",

            "-use_wallclock_as_timestamps", "1",

            "-i", rtsp_url,

            "-vf", "scale=1280:720",

            "-f", "rawvideo",
            "-pix_fmt", "bgr24",

            "-an",
            "-sn",

            "-"
        ]

        self.start_ffmpeg()

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()

    def start_ffmpeg(self):

        self.pipe = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

    def reconnect(self):

        print("[INFO] Reconnecting camera...")

        try:
            self.pipe.kill()
        except:
            pass

        time.sleep(1)

        self.start_ffmpeg()

    def update(self):

        while self.running:

            try:

                raw_frame = self.pipe.stdout.read(
                    self.frame_size
                )

                if len(raw_frame) != self.frame_size:

                    print("[ERROR] Camera disconnected...")
                    self.reconnect()
                    continue

                frame = np.frombuffer(
                    raw_frame,
                    dtype=np.uint8
                )

                frame = frame.reshape(
                    (self.HEIGHT, self.WIDTH, 3)
                )

                # Store ONLY latest frame
                self.latest_frame = frame

            except Exception as e:

                print(f"[ERROR] {e}")

                self.reconnect()

    def read(self):

        if self.latest_frame is None:
            return False, None

        return True, self.latest_frame.copy()

    def release(self):

        self.running = False

        try:
            self.pipe.kill()
        except:
            pass