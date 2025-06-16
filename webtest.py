# 주요 기능:
# - 1개 카메라에서 프레임 읽기
# - 사람 수 체크: 1명만 10초 이상 감지되면 경고
# - 안전모 미착용: 5초 이상 감지 안 되면 경고
# - 경고 횟수 카운트 및 실시간 알림 (화면 고정)

import cv2
import torch
import threading
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pyngrok import ngrok
from ultralytics import YOLO
from queue import Queue
import pygame
from fastapi.staticfiles import StaticFiles
import time

app = FastAPI()
app.mount("/web", StaticFiles(directory="web"), name="web")

# 모델 로딩
person_model = YOLO("yolo11n.pt")
helmet_model = YOLO("customhardhat_v1.2.pt")

frame_queue = Queue()
event_queue = Queue()
alarm_playing = threading.Event()

# 경고 횟수 count 위한 변수 선언언
helmet_missing_start = None
alone_start = None
helmet_alert_count = 0
alone_alert_count = 0

pygame.mixer.init()

# 알람 재생 위한 함수 선언
def play_alarm(sound_file):
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()

# 카메라 실행 위한 함수 선언
def capture_frames(cam_id: int):
    global helmet_missing_start, alone_start, helmet_alert_count, alone_alert_count
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        person_results = person_model(frame, conf=0.5, verbose=False)
        helmet_results = helmet_model(frame, conf=0.5, verbose=False)

        persons = 0
        helmets = 0
        now = time.time()

        for r in person_results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                if cls == 0:
                    persons += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        for r in helmet_results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                if cls == 0:
                    helmets += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
                    cv2.putText(frame, "Helmet", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        if persons > 0 and helmets == 0:
            if helmet_missing_start is None:
                helmet_missing_start = now
            elif now - helmet_missing_start >= 5:
                helmet_alert_count += 1
                event_queue.put({"type": "helmet", "message": helmet_alert_count})
                threading.Thread(target=play_alarm, args=("hardhat.mp3",), daemon=True).start()
                helmet_missing_start = now
        else:
            helmet_missing_start = None

        if persons == 1:
            if alone_start is None:
                alone_start = now
            elif now - alone_start >= 10:
                alone_alert_count += 1
                event_queue.put({"type": "alone", "message": alone_alert_count})
                threading.Thread(target=play_alarm, args=("personnumber.mp3",), daemon=True).start()
                alone_start = now
        else:
            alone_start = None

        frame_queue.put(frame)

    cap.release()

@app.get("/")
async def home():
    return HTMLResponse("""
    <html><head><style>
    body { display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: sans-serif; }
    h1 { text-align: center; }
    #alerts { display: flex; gap: 20px; margin-top: 20px; }
    .alertBox {
        background: #f33; color: white; padding: 10px 20px;
        border-radius: 5px; font-weight: bold; font-size: 18px;
    }
    canvas { margin-top: 10px; border: 2px solid black; }
    </style></head>
    <body>
    <h1>위험 모니터링</h1>
    <canvas id='view' width='640' height='480'></canvas>
    <div id='alerts'>
        <div class="alertBox" id="helmetAlert">🚨 안전모 미착용: 누적 0회</div>
        <div class="alertBox" id="aloneAlert">⚠️ 1인 근무 감지: 누적 0회</div>
    </div>
    <script>
    const canvas = document.getElementById('view');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); };
    const ws = new WebSocket(`wss://${location.host}/video/live/0`);
    ws.binaryType = 'blob';
    ws.onmessage = e => { const blob = new Blob([e.data]); img.src = URL.createObjectURL(blob); };

    const alertWS = new WebSocket(`wss://${location.host}/event/0`);
    alertWS.onmessage = e => {
        const d = JSON.parse(e.data);
        if (d.type === "helmet") {
            document.getElementById("helmetAlert").textContent = `안전모 미착용: 누적 ${d.message}회`;
        } else if (d.type === "alone") {
            document.getElementById("aloneAlert").textContent = `1인 근무 감지: 누적 ${d.message}회`;
        }
    }
    </script></body></html>
    """)

@app.websocket("/video/live/{cam_id}")
async def video_stream(websocket: WebSocket, cam_id: int):
    await websocket.accept()
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                await websocket.send_bytes(jpeg.tobytes())
        await asyncio.sleep(0.03)

@app.websocket("/event/{cam_id}")
async def event_stream(websocket: WebSocket, cam_id: int):
    await websocket.accept()
    try:
        while True:
            if not event_queue.empty():
                event = event_queue.get()
                await websocket.send_json(event)
            else:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(5000)
    print(f"Public URL: {ngrok_tunnel.public_url}")
    threading.Thread(target=capture_frames, args=(0,), daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)