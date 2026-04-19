import requests, base64, cv2
img = cv2.imread("lena.jpg")
_, buffer = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buffer).decode('utf-8')
res = requests.post("http://127.0.0.1:8001/attendance/mark", json={"image": f"data:image/jpeg;base64,{b64}"})
print("STATUS:", res.status_code)
print("BODY:", res.text)
