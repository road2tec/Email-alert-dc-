import requests, base64, cv2

# Get an actual face image
img_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
image_path = "lena.jpg"
with open(image_path, 'wb') as f:
    f.write(requests.get(img_url).content)

img = cv2.imread(image_path)
_, buffer = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buffer).decode('utf-8')

res = requests.post("http://127.0.0.1:8001/face/recognize", json={"image": f"data:image/jpeg;base64,{b64}"})
print(res.status_code)
print(res.text)
