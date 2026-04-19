import subprocess
import requests
import time
import base64
import cv2

p = subprocess.Popen(["./venv/bin/python", "-m", "uvicorn", "backend.app:app", "--port", "8002"], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

time.sleep(3) # allow startup

# Get an actual face image
img = cv2.imread("lena.jpg")
_, buffer = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buffer).decode('utf-8')

try:
    res = requests.post("http://127.0.0.1:8002/face/recognize", json={"image": f"data:image/jpeg;base64,{b64}"})
    print("STATUS:", res.status_code)
except Exception as e:
    print("REQUEST FAILED:", e)

p.terminate()
outs, errs = p.communicate()
print("---- STDOUT ----\n", outs)
print("---- STDERR ----\n", errs)
