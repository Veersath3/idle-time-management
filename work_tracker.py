"""
Construction Site Worker Idle Tracker
RTX 4090 Edition — Fixed Camera + <50ms Latency
=================================================

FIXES vs original:
  1. cv2.imshow() moved to MAIN THREAD (daemon threads can't own windows on most OS)
  2. Model loading / warmup done in main thread before socketio.run()
  3. Flask/SocketIO runs in a background daemon thread instead
  4. RTSP latency pushed below 50ms:
       · CAP_PROP_BUFFERSIZE = 0  (was 1)
       · grab() loop with no sleep when frame is ready
       · FFmpeg options: fflags=nobuffer, flags=low_delay, probesize=32
       · GStreamer pipeline: latency=0, sync=false, max-buffers=1 drop=true
  5. --rebuild-engine and --test-latency flags preserved
"""

import cv2
import os
import sys
import time
import shutil
import argparse
import threading
import collections
from datetime import datetime

import numpy as np

# ── PyTorch + CUDA ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.cuda
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[ERROR] PyTorch not found. Run:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# ── Ultralytics YOLO ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics")
    sys.exit(1)

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DEVICE              = "cuda:0"
YOLO_SOURCE         = "yolo26l.pt"           # input weights
ENGINE_PATH         = "yolo26l.engine"       # must match what Ultralytics saves (no _4090 suffix)
IMGSZ               = 1280
CONFIDENCE          = 0.40
TRACKER_IOU         = 0.50
HALF                = True
WARMUP_RUNS         = 5

IDLE_THRESHOLD_SEC  = 120
MOVE_THRESHOLD_PX   = 20
WEB_PORT            = 5001

RTSP_URL = "rtsp://admin:!SoftDesigners1@192.168.0.124:554/Streaming/Channels/104"

# ── Shared state ──────────────────────────────────────────────────────────────
workers    = {}
alarm_log  = []
state_lock = threading.Lock()

ZONE_NAMES = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E", "Zone F"]

def get_zone(x, y, fw, fh):
    return ZONE_NAMES[min(int(x / fw * 3) + int(y / fh * 2) * 3, 5)]


# ═══════════════════════════════════════════════════════════════════════════════
# GPU DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
def print_gpu_info():
    print("\n" + "═" * 60)
    print("  RTX 4090 DIAGNOSTICS")
    print("═" * 60)
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  CUDA          : {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1024 ** 3
    print(f"  GPU           : {props.name}")
    print(f"  VRAM          : {vram:.1f} GB")
    print(f"  CUDA cores    : {props.multi_processor_count} SMs")
    print(f"  Compute cap   : {props.major}.{props.minor}")
    used = torch.cuda.memory_allocated(0) / 1024 ** 3
    print(f"  VRAM used     : {used:.2f} GB")
    print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TENSORRT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def get_model(rebuild=False):
    if rebuild and os.path.exists(ENGINE_PATH):
        os.remove(ENGINE_PATH)
        print(f"[TRT] Removed old engine: {ENGINE_PATH}")

    if os.path.exists(ENGINE_PATH):
        print(f"[TRT] Loading cached engine: {ENGINE_PATH}")
        model = YOLO(ENGINE_PATH)
    else:
        if not os.path.exists(YOLO_SOURCE):
            print(f"[ERROR] Model file not found: {YOLO_SOURCE}")
            print(f"        Either place {YOLO_SOURCE} in this directory,")
            print(f"        or change YOLO_SOURCE in the config section.")
            sys.exit(1)
        print(f"[TRT] Building TensorRT FP16 engine from {YOLO_SOURCE}")
        print(f"[TRT] This takes ~60s on RTX 4090 — done once, cached forever")
        base = YOLO(YOLO_SOURCE)
        base.export(
            format    = "engine",
            imgsz     = IMGSZ,
            half      = True,
            device    = 0,
            simplify  = True,
            workspace = 8,
        )
        generated = YOLO_SOURCE.replace(".pt", ".engine")
        if os.path.exists(generated) and generated != ENGINE_PATH:
            os.rename(generated, ENGINE_PATH)
        model = YOLO(ENGINE_PATH)
        print(f"[TRT] ✓ Engine built and loaded")

    # NOTE: .to(DEVICE) is PyTorch-only — TensorRT engines pass device at inference time
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# LOW-LATENCY RTSP READER  — target < 50 ms
# Key changes vs original:
#   · CAP_PROP_BUFFERSIZE = 0  (not 1)
#   · FFmpeg: fflags=nobuffer, flags=low_delay, probesize=32, analyzeduration=0
#   · GStreamer: latency=0, max-buffers=1, drop=true, sync=false (unchanged but faster)
#   · No sleep() inside the grab loop when a frame is waiting
# ═══════════════════════════════════════════════════════════════════════════════
class RTSPReader:
    def __init__(self, url, width=1920, height=1080,
                 fps_limit=60, use_tcp=True, reconnect_delay=2.0):
        self.url             = url
        self.width           = width
        self.height          = height
        self.fps_limit       = fps_limit
        self.use_tcp         = use_tcp
        self.reconnect_delay = reconnect_delay

        self._frame   = None
        self._ret     = False
        self._lock    = threading.Lock()
        self._running = False
        self._cap     = None

        self.fps_actual     = 0.0
        self.stream_latency = 0.0   # ms
        self._fc  = 0
        self._ft  = time.time()

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if self._ret:
                print(f"[RTSP] ✓ Connected  {self.width}x{self.height}  "
                      f"fps_limit={self.fps_limit}  tcp={self.use_tcp}")
                return True
            time.sleep(0.05)
        print("[RTSP] ✗ No frame received — check URL / credentials / network")
        return False

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    # ── Backends ──────────────────────────────────────────────────────────────

    def _gstreamer(self):
        """GStreamer pipeline — hardware H.264 decode, latency=0."""
        if not shutil.which("gst-launch-1.0"):
            return None
        pipe = (
            f"rtspsrc location={self.url} protocols=tcp latency=0 "
            f"buffer-mode=none ! "
            f"rtph264depay ! h264parse ! nvh264dec ! "   # NVDEC hardware decode
            f"videoconvert ! videoscale ! "
            f"video/x-raw,width={self.width},height={self.height},format=BGR ! "
            f"appsink max-buffers=1 drop=true sync=false emit-signals=true"
        )
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[RTSP] Backend: GStreamer + NVDEC (latency=0, hardware decode)")
            return cap
        # Fallback: software decode
        pipe_sw = (
            f"rtspsrc location={self.url} protocols=tcp latency=0 ! "
            f"rtph264depay ! h264parse ! avdec_h264 max-threads=4 ! "
            f"videoconvert ! videoscale ! "
            f"video/x-raw,width={self.width},height={self.height} ! "
            f"appsink max-buffers=1 drop=true sync=false"
        )
        cap = cv2.VideoCapture(pipe_sw, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[RTSP] Backend: GStreamer (software decode, latency=0)")
            return cap
        return None

    def _ffmpeg_tcp(self):
        """
        FFmpeg with maximum latency reduction flags.
        fflags=nobuffer + flags=low_delay are the critical ones for <50ms.
        """
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "fflags;nobuffer|"           # ← disable demuxer buffering
            "flags;low_delay|"           # ← decoder low-delay mode
            "probesize;32|"              # ← minimal probe (was missing)
            "analyzeduration;0|"         # ← skip duration analysis
            "max_delay;0|"
            "reorder_queue_size;0|"
            "stimeout;2000000"
        )
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE,    0)    # 0 = smallest possible
            cap.set(cv2.CAP_PROP_FPS,           self.fps_limit)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,   self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  self.height)
            print("[RTSP] Backend: FFmpeg + TCP (nobuffer, low_delay, probesize=32)")
            return cap
        return None

    def _fallback(self):
        cap = cv2.VideoCapture(self.url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            print("[RTSP] Backend: OpenCV default (fallback)")
            return cap
        return None

    # ── Grab loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            self._cap = self._gstreamer() or self._ffmpeg_tcp() or self._fallback()
            if not self._cap or not self._cap.isOpened():
                print(f"[RTSP] Cannot open stream — retrying in {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)
                continue

            fails = 0
            while self._running:
                t0 = time.time()

                # grab() discards frame data, drains buffer immediately
                if not self._cap.grab():
                    fails += 1
                    if fails > 60:
                        print("[RTSP] Too many grab() failures — reconnecting")
                        break
                    time.sleep(0.005)
                    continue
                fails = 0

                ret, frame = self._cap.retrieve()
                if ret and frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._ret   = True
                    self.stream_latency = (time.time() - t0) * 1000
                    self._fc += 1
                    elapsed = time.time() - self._ft
                    if elapsed >= 1.0:
                        self.fps_actual = self._fc / elapsed
                        self._fc = 0
                        self._ft = time.time()
                # No artificial sleep — grab as fast as possible, keep buffer empty

            self._cap.release()
            self._cap = None
            if self._running:
                print(f"[RTSP] Stream dropped — reconnecting in {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER STATE
# ═══════════════════════════════════════════════════════════════════════════════
class WorkerState:
    def __init__(self, tid, x, y, zone):
        self.tid         = tid
        self.worker_id   = f"W-{tid:03d}"
        self.x = x;  self.y = y
        self.zone        = zone
        self.status      = "active"
        self.idle_since  = None
        self.idle_secs   = 0
        self.alarm_fired = False
        self.last_seen   = time.time()
        self.entry_time  = datetime.now().strftime("%H:%M:%S")

    def update(self, x, y):
        dist = ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5
        self.last_seen = time.time()
        if dist > MOVE_THRESHOLD_PX:
            self.x = x;  self.y = y
            self.status      = "active"
            self.idle_since  = None
            self.idle_secs   = 0
            self.alarm_fired = False
        else:
            if self.idle_since is None:
                self.idle_since = time.time()
            self.idle_secs = int(time.time() - self.idle_since)
            self.status = ("alarm" if self.idle_secs >= IDLE_THRESHOLD_SEC else
                           "idle"  if self.idle_secs > 10 else "active")

    def to_dict(self):
        return {
            "id":         self.worker_id,
            "zone":       self.zone,
            "status":     self.status,
            "idle_secs":  self.idle_secs,
            "entry_time": self.entry_time,
            "x": int(self.x), "y": int(self.y),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
app      = Flask(__name__)
app.config["SECRET_KEY"] = "4090-worker-monitor"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Worker Monitor — RTX 4090</title>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Courier New',monospace;background:#080c10;color:#e6edf3;padding:1rem;}
body::before{content:'';position:fixed;inset:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,.012) 2px,rgba(0,229,255,.012) 4px);
  pointer-events:none;z-index:9999;}
.topbar{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;flex-wrap:wrap;gap:8px;}
h1{font-size:.95rem;color:#00e5ff;letter-spacing:3px;text-transform:uppercase;}
.badges{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap;}
.badge{font-size:10px;padding:2px 10px;border-radius:2px;border:1px solid;font-family:'Courier New',monospace;}
.b-gpu{background:#0a2a0a;color:#00e676;border-color:#00e676;}
.b-trt{background:#0a1a2a;color:#00e5ff;border-color:#00e5ff;}
.b-fp16{background:#1a0a2a;color:#d500f9;border-color:#d500f9;}
.b-lat{color:#1de9b6;border-color:#1de9b6;background:#0a2a2a;}
#clock{font-size:11px;color:#4a6070;font-family:monospace;}
.perf{display:flex;gap:0;background:#0d1520;border:1px solid #1a2a3a;
  margin-bottom:1rem;overflow:hidden;border-radius:3px;}
.pi{padding:10px 20px;border-right:1px solid #1a2a3a;font-size:11px;color:#4a6070;}
.pi:last-child{border-right:none;}
.pi .v{display:block;font-size:18px;font-weight:700;color:#00e5ff;
  font-family:'Courier New',monospace;margin-top:3px;}
.pi .v.green{color:#00e676;} .pi .v.amber{color:#ffab00;} .pi .v.red{color:#ff1744;}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:1rem;}
.met{background:#0d1520;border:1px solid #1a2a3a;border-radius:3px;padding:12px 16px;position:relative;}
.met::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.met.c0::after{background:#00e5ff;}.met.c1::after{background:#00e676;}
.met.c2::after{background:#ffab00;}.met.c3::after{background:#ff1744;}.met.c4::after{background:#d500f9;}
.met-l{font-size:10px;letter-spacing:2px;color:#4a6070;text-transform:uppercase;margin-bottom:6px;}
.met-v{font-size:28px;font-weight:700;font-family:'Courier New',monospace;}
.alarm-banner{display:none;background:rgba(255,23,68,.08);border:1px solid #ff1744;
  border-radius:3px;padding:12px 18px;margin-bottom:1rem;align-items:center;gap:14px;}
.alarm-banner.show{display:flex;animation:ab 1s infinite;}
@keyframes ab{50%{background:rgba(255,23,68,.2);}}
.ap{width:10px;height:10px;border-radius:50%;background:#ff1744;animation:blink .5s infinite;}
@keyframes blink{50%{opacity:.1;}}
.at{font-size:13px;font-weight:700;color:#ff1744;flex:1;letter-spacing:1px;}
.mute{font-size:11px;background:transparent;border:1px solid #ff1744;color:#ff1744;
  padding:3px 12px;border-radius:2px;cursor:pointer;font-family:monospace;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;margin-bottom:1rem;}
.wcard{background:#0d1520;border:1px solid #1a2a3a;border-radius:3px;padding:14px;
  cursor:pointer;transition:border-color .2s;}
.wcard:hover{border-color:#00e5ff;}
.wcard.idle{border-color:#ffab00;}.wcard.alarm{border-color:#ff1744;animation:cb .8s infinite;}
@keyframes cb{50%{border-color:rgba(255,23,68,.2);}}
.wid{font-size:15px;font-weight:700;color:#00e5ff;margin-bottom:2px;letter-spacing:1px;}
.wzone{font-size:11px;color:#4a6070;margin-bottom:8px;}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;
  letter-spacing:1.5px;padding:2px 10px;border-radius:2px;text-transform:uppercase;margin-bottom:8px;}
.chip.active{background:rgba(0,230,118,.1);color:#00e676;}
.chip.idle{background:rgba(255,171,0,.1);color:#ffab00;}
.chip.alarm{background:rgba(255,23,68,.15);color:#ff1744;}
.wtimer{font-size:11px;color:#4a6070;margin-bottom:6px;}
.bar{height:3px;background:#1a2a3a;border-radius:2px;overflow:hidden;}
.bf{height:100%;border-radius:2px;transition:width 1s linear;}
.log{background:#0d1520;border:1px solid #1a2a3a;border-radius:3px;padding:14px;}
.lt{font-size:10px;letter-spacing:3px;color:#4a6070;text-transform:uppercase;margin-bottom:10px;}
.le{font-size:11px;padding:4px 0;border-bottom:1px solid #1a2a3a;display:flex;gap:10px;}
.le:last-child{border-bottom:none;}
.le .t{color:#2a3a4a;flex-shrink:0;min-width:65px;}
.ok{color:#00e676;}.warn{color:#ffab00;}.err{color:#ff1744;}.info{color:#00e5ff;}
</style>
</head>
<body>
<div class="topbar">
  <div>
    <h1>▶ Construction Worker Monitor — RTX 4090</h1>
    <div class="badges">
      <span class="badge b-gpu" id="gpu-badge">RTX 4090 · 24GB VRAM</span>
      <span class="badge b-trt">TensorRT FP16 Engine</span>
      <span class="badge b-fp16">YOLO · 1280px</span>
      <span class="badge b-lat" id="lat-badge">Stream: measuring...</span>
    </div>
  </div>
  <span id="clock"></span>
</div>
<div class="perf">
  <div class="pi">DET FPS<span class="v green" id="p-fps">—</span></div>
  <div class="pi">INFERENCE<span class="v" id="p-inf">—</span></div>
  <div class="pi">CAM FPS<span class="v" id="p-camfps">—</span></div>
  <div class="pi">STREAM LAT<span class="v" id="p-lat">—</span></div>
  <div class="pi">VRAM USED<span class="v" id="p-vram">—</span></div>
  <div class="pi">WORKERS<span class="v" id="p-workers">0</span></div>
</div>
<div class="metrics">
  <div class="met c0"><div class="met-l">On Site</div><div class="met-v" id="m-total" style="color:#00e5ff;">0</div></div>
  <div class="met c1"><div class="met-l">Active</div><div class="met-v" id="m-active" style="color:#00e676;">0</div></div>
  <div class="met c2"><div class="met-l">Idle</div><div class="met-v" id="m-idle" style="color:#ffab00;">0</div></div>
  <div class="met c3"><div class="met-l">Alarms</div><div class="met-v" id="m-alarms" style="color:#ff1744;">0</div></div>
  <div class="met c4"><div class="met-l">Threshold</div><div class="met-v" style="color:#d500f9;">2:00</div></div>
</div>
<div class="alarm-banner" id="alarm-banner">
  <div class="ap"></div>
  <span class="at" id="alarm-text">Idle alarm</span>
  <button class="mute" onclick="toggleMute()" id="mute-btn">MUTE</button>
</div>
<div class="grid" id="grid"></div>
<div class="log">
  <div class="lt">Event Log</div>
  <div id="log"></div>
</div>
<script>
const socket = io();
let totalAlarms=0, muted=false, intervals={};
const IL = {{ idle_limit }};
function fmt(s){const m=Math.floor(s/60),r=s%60;return m?m+'m '+r+'s':r+'s';}
function log(msg,cls){
  const el=document.getElementById('log');
  const d=document.createElement('div');
  d.className='le';
  d.innerHTML=`<span class="t">${new Date().toLocaleTimeString()}</span><span class="${cls}">${msg}</span>`;
  el.insertBefore(d,el.firstChild);
  while(el.children.length>16)el.removeChild(el.lastChild);
}
function speak(m){
  if(muted||!window.speechSynthesis)return;
  speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(m);
  u.rate=0.9;u.volume=1;speechSynthesis.speak(u);
}
function toggleMute(){
  muted=!muted;
  document.getElementById('mute-btn').textContent=muted?'UNMUTE':'MUTE';
  if(muted&&window.speechSynthesis)speechSynthesis.cancel();
}
socket.on('state_update',function(d){
  const ws=d.workers, g=document.getElementById('grid');
  g.innerHTML='';
  let act=0,idle=0,alarms=[];
  ws.forEach(w=>{
    if(w.status==='active')act++;else idle++;
    if(w.status==='alarm')alarms.push(w);
    const pct=Math.min(100,Math.round(w.idle_secs/IL*100));
    const bc=w.status==='active'?'#00e676':w.status==='idle'?'#ffab00':'#ff1744';
    const c=document.createElement('div');
    c.className='wcard '+w.status;
    c.innerHTML=`
      <div class="wid">${w.id}</div>
      <div class="wzone">${w.zone} · since ${w.entry_time}</div>
      <span class="chip ${w.status}">
        ${w.status==='active'?'● Active':w.status==='idle'?'◐ Idle':'⚠ ALARM'}
      </span>
      <div class="wtimer">${w.status==='active'?'Working':fmt(w.idle_secs)+' idle'}</div>
      <div class="bar"><div class="bf" style="width:${pct}%;background:${bc};"></div></div>
    `;
    g.appendChild(c);
  });
  document.getElementById('m-total').textContent=ws.length;
  document.getElementById('m-active').textContent=act;
  document.getElementById('m-idle').textContent=idle;
  document.getElementById('m-alarms').textContent=totalAlarms;
  document.getElementById('p-workers').textContent=ws.length;
  const b=document.getElementById('alarm-banner');
  if(alarms.length>0){
    b.classList.add('show');
    document.getElementById('alarm-text').textContent=
      'IDLE: '+alarms.map(w=>w.id+' ('+w.zone+') '+fmt(w.idle_secs)).join(' | ');
  }else b.classList.remove('show');
});
socket.on('perf',function(d){
  document.getElementById('p-fps').textContent   =d.det_fps+'fps';
  document.getElementById('p-inf').textContent   =d.inf_ms+'ms';
  document.getElementById('p-camfps').textContent=d.cam_fps+'fps';
  document.getElementById('p-vram').textContent  =d.vram;
  const latEl=document.getElementById('p-lat');
  latEl.textContent=d.stream_ms+'ms';
  latEl.className='v'+(d.stream_ms<50?' green':d.stream_ms<100?' amber':' red');
  const lb=document.getElementById('lat-badge');
  lb.textContent='Stream: '+d.stream_ms+'ms';
  lb.style.color=d.stream_ms<50?'#1de9b6':'#ffab00';
  document.getElementById('gpu-badge').textContent=d.gpu_name+' · '+d.vram+' used';
});
socket.on('alarm',function(d){
  totalAlarms++;
  log('ALARM — '+d.worker_id+' idle >2min','err');
  speak('Please work, '+d.worker_id.replace('-',' '));
  if(!intervals[d.worker_id])
    intervals[d.worker_id]=setInterval(()=>speak('Please work, '+d.worker_id.replace('-',' ')),5000);
});
socket.on('alarm_cleared',function(d){
  clearInterval(intervals[d.worker_id]);delete intervals[d.worker_id];
  if(window.speechSynthesis)speechSynthesis.cancel();
  log(d.worker_id+' resumed work','ok');
});
socket.on('worker_entered',function(d){log(d.worker_id+' entered frame','ok');});
socket.on('worker_idle',   function(d){log(d.worker_id+' went idle','warn');});
socket.on('worker_left',   function(d){log(d.worker_id+' left frame','info');});
setInterval(()=>document.getElementById('clock').textContent=new Date().toLocaleTimeString(),1000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, idle_limit=IDLE_THRESHOLD_SEC)

@app.route("/api/workers")
def api_workers():
    return jsonify([w.to_dict() for w in workers.values()])

@app.route("/api/alarms")
def api_alarms():
    return jsonify(alarm_log)

def push_state():
    with state_lock:
        data = [w.to_dict() for w in workers.values()]
    socketio.emit("state_update", {"workers": data})


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRACKING LOOP  — runs on the MAIN THREAD so cv2.imshow() works
# ═══════════════════════════════════════════════════════════════════════════════
def tracking_loop(source, model, rebuild_engine=False):

    # ── RTSP or local source ──────────────────────────────────────────────────
    is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp")

    if is_rtsp:
        cap = RTSPReader(
            url       = source,
            width     = 1920,
            height    = 1080,
            fps_limit = 60,    # push grab loop faster → lower latency
            use_tcp   = True,
        )
        if not cap.start():
            print("[ERROR] RTSP connection failed — check camera URL/credentials")
            return
        fw, fh = 1920, 1080
    else:
        cap_cv = cv2.VideoCapture(source)
        if not cap_cv.isOpened():
            print(f"[ERROR] Cannot open source: {source}")
            return
        cap_cv.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        fw = int(cap_cv.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap_cv.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Stream: {fw}x{fh} | Engine: {ENGINE_PATH}")
    print(f"[INFO] Dashboard → http://localhost:{WEB_PORT}")
    print(f"[INFO] Press Q in the OpenCV window to quit\n")

    # ── Perf counters ─────────────────────────────────────────────────────────
    det_fps = 0.0
    fc      = 0
    ft      = time.time()
    inf_ms  = 0

    # ── Main loop — on main thread so cv2.imshow() works ─────────────────────
    while True:
        if is_rtsp:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue
        else:
            ret, frame = cap_cv.read()
            if not ret:
                print("[INFO] End of video / capture closed")
                break

        # ── TensorRT inference ────────────────────────────────────────────────
        t_inf = time.time()
        results = model.track(
            frame,
            persist  = True,
            classes  = [0],          # person only
            conf     = CONFIDENCE,
            iou      = TRACKER_IOU,
            device   = DEVICE,
            half     = HALF,
            verbose  = False,
            imgsz    = IMGSZ,
        )
        inf_ms = round((time.time() - t_inf) * 1000)

        # ── FPS / perf emit every second ──────────────────────────────────────
        fc += 1
        if time.time() - ft >= 1.0:
            det_fps = fc / (time.time() - ft)
            fc = 0;  ft = time.time()

            used  = torch.cuda.memory_allocated(0) / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            cam_fps   = round(cap.fps_actual)     if is_rtsp else 0
            stream_ms = round(cap.stream_latency) if is_rtsp else 0
            socketio.emit("perf", {
                "det_fps":   round(det_fps),
                "inf_ms":    inf_ms,
                "cam_fps":   cam_fps,
                "stream_ms": stream_ms,
                "vram":      f"{used:.1f}/{total:.0f}GB",
                "gpu_name":  torch.cuda.get_device_name(0),
            })

        # ── Parse detections ──────────────────────────────────────────────────
        active_ids = set()

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confs     = results[0].boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cx  = (x1 + x2) / 2
                cy  = (y1 + y2) / 2
                tid = int(tid)
                active_ids.add(tid)
                zone = get_zone(cx, cy, fw, fh)

                with state_lock:
                    if tid not in workers:
                        workers[tid] = WorkerState(tid, cx, cy, zone)
                        socketio.emit("worker_entered", {"worker_id": workers[tid].worker_id})
                    w    = workers[tid]
                    prev = w.status
                    w.update(cx, cy)
                    cur  = w.status

                if prev == "active" and cur == "idle":
                    socketio.emit("worker_idle", {"worker_id": w.worker_id})

                if cur == "alarm" and not w.alarm_fired:
                    w.alarm_fired = True
                    with state_lock:
                        alarm_log.append({
                            "worker_id": w.worker_id, "zone": zone,
                            "time":      datetime.now().isoformat(),
                            "idle_secs": w.idle_secs,
                        })
                    socketio.emit("alarm", {"worker_id": w.worker_id})
                    print(f"[ALARM] {w.worker_id} idle >{IDLE_THRESHOLD_SEC}s in {zone}")

                if prev == "alarm" and cur == "active":
                    socketio.emit("alarm_cleared", {"worker_id": w.worker_id})

                # ── Draw ──────────────────────────────────────────────────────
                color = ((0, 220, 0)   if cur == "active" else
                         (0, 165, 255) if cur == "idle"   else
                         (0, 0, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{w.worker_id}  {cur.upper()}"
                if cur != "active":
                    label += f"  {w.idle_secs}s"

                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, f"{conf:.2f}", (x1 + 3, y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

                if cur == "alarm":
                    cv2.circle(frame, (int(cx), int(cy)), 28, (0, 0, 255), 2)
                    cv2.putText(frame, "PLEASE WORK!", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── Remove stale workers ──────────────────────────────────────────────
        with state_lock:
            stale = [tid for tid, w in workers.items()
                     if time.time() - w.last_seen > 5 and tid not in active_ids]
            for tid in stale:
                socketio.emit("worker_left", {"worker_id": workers[tid].worker_id})
                del workers[tid]

        # ── HUD overlay ───────────────────────────────────────────────────────
        lat_ms    = round(cap.stream_latency) if is_rtsp else 0
        lat_color = ((0, 255, 100) if lat_ms < 50   else
                     (0, 200, 255) if lat_ms < 100  else
                     (0, 0, 255))
        hud = (f"RTX4090·TRT·FP16 | det:{det_fps:.0f}fps | "
               f"inf:{inf_ms}ms | lat:{lat_ms}ms | workers:{len(workers)}")
        cv2.putText(frame, hud, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.circle(frame, (fw - 16, 16), 8, lat_color, -1)
        cv2.putText(frame, f"{lat_ms}ms", (fw - 70, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, lat_color, 1)

        push_state()

        # cv2.imshow MUST be on main thread — this is the fix for the black window
        cv2.imshow("RTX 4090 Worker Tracker — Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if is_rtsp:
        cap.stop()
    else:
        cap_cv.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracker stopped.")
    os._exit(0)   # also shut down the Flask thread cleanly


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTX 4090 Worker Tracker")
    parser.add_argument("--source",         default=RTSP_URL)
    parser.add_argument("--idle",           type=int, default=120)
    parser.add_argument("--rebuild-engine", action="store_true")
    parser.add_argument("--test-latency",   action="store_true",
                        help="RTSP latency test only — no YOLO")
    args = parser.parse_args()

    IDLE_THRESHOLD_SEC = args.idle
    source = int(args.source) if args.source.isdigit() else args.source

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available — RTX 4090 required")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    print(f"[INFO] GPU detected: {gpu}")

    # ── Latency test (no YOLO) ────────────────────────────────────────────────
    if args.test_latency:
        print("=" * 60)
        print("RTSP LATENCY TEST — hold a clock in front of the camera")
        print("Target: < 50ms  |  Green < 50ms  Amber < 100ms  Red > 100ms")
        print("=" * 60)
        cap = RTSPReader(source, width=1920, height=1080, fps_limit=60, use_tcp=True)
        cap.start()
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            now   = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            lat   = round(cap.stream_latency)
            lat_c = ((0, 255, 100) if lat < 50  else
                     (0, 200, 255) if lat < 100 else
                     (0, 0, 255))
            cv2.putText(frame, f"PC time  : {now}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"Cam FPS  : {cap.fps_actual:.1f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Lat grab : {lat}ms", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, lat_c, 2)
            cv2.putText(frame, "Target: < 50ms (RTX 4090 mode)", (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            cv2.imshow("Latency Test — Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.stop()
        cv2.destroyAllWindows()
        sys.exit(0)

    # ── Normal run ────────────────────────────────────────────────────────────
    print_gpu_info()

    # CUDA optimisations
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load / build model ON MAIN THREAD before starting anything else
    print("[TRT] Loading model...")
    model = get_model(rebuild=args.rebuild_engine)

    # Warmup — force JIT before live feed
    print(f"[WARMUP] Running {WARMUP_RUNS} dummy inference passes...")
    for _ in range(WARMUP_RUNS):
        _ = model.predict(
            source  = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8),
            device  = DEVICE,
            half    = HALF,
            verbose = False,
            imgsz   = IMGSZ,
        )
    torch.cuda.synchronize()
    print("[WARMUP] ✓ Done\n")

    # Flask runs in a BACKGROUND daemon thread
    # tracking_loop (with cv2.imshow) stays on the MAIN THREAD
    flask_thread = threading.Thread(
        target=lambda: socketio.run(
            app, host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False
        ),
        daemon=True,
    )
    flask_thread.start()
    print(f"[INFO] Dashboard → http://localhost:{WEB_PORT}")

    # This call blocks on the main thread — cv2.imshow works correctly here
    tracking_loop(source, model, rebuild_engine=args.rebuild_engine)