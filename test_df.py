import cv2, traceback
try:
    from deepface import DeepFace
    img = cv2.imread("lena.jpg")
    print("Executing DeepFace.represent...")
    emb = DeepFace.represent(img_path=img, model_name="Facenet", detector_backend="skip", enforce_detection=False)
    print("Embedding size:", len(emb[0]["embedding"]))
except Exception as e:
    print("TRACEBACK:")
    traceback.print_exc()
