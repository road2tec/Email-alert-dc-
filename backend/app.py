import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import base64
import shutil
import threading
from typing import List
from datetime import datetime
from pymongo import MongoClient
import glob
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()
# ==== CONFIG ====
UPLOAD_DIR = "public/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
STUDENT_IMAGES_DIR = "backend/Student_Images"

app = FastAPI()

# Mount public directory for uploads
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

from .database import students_collection, attendance_collection, admin_collection, MONGO_URI
SIMILARITY_THRESHOLD = 0.65 

# ==== MEDIAPIPE LAZY SETUP ====
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

face_detection_model = None

def get_face_detector():
    global face_detection_model
    if face_detection_model is None:
        print("[INFO] Initializing MediaPipe Face Detection (Tasks API)...")
        base_options = mp_python.BaseOptions(model_asset_path='backend/blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.15)
        face_detection_model = vision.FaceDetector.create_from_options(options)
    return face_detection_model

# ==== GLOBAL STATE ====
known_faces = [] # List of {"name": name, "hist": histogram}

print(f"[INFO] Connected to MongoDB at {MONGO_URI}")

# ==== CORS ====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== HELPER FUNCTIONS ====
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

last_unknown_alert_time = 0
ALERT_COOLDOWN_SECONDS = 30

def send_unknown_alert_email():
    try:
        SMTP_EMAIL = os.getenv("SMTP_EMAIL", "projectiot1406@gmail.com")
        SMTP_PASS = os.getenv("SMTP_PASS", "xyxjtekycznhpxyo")
        ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", "punamchanne51@gmail.com")

        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = ALERT_RECEIVER
        msg['Subject'] = "🚨 SECURITY ALERT: Unauthorized Person Detected"

        time_str = datetime.now().strftime('%d %b %Y, %I:%M %p')
        html_body = f"""
        <html>
            <body style="font-family: 'Inter', Arial, sans-serif; background-color: #f4f4f9; padding: 20px; color: #333; margin: 0;">
                <div style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                    <div style="background-color: #d32f2f; color: #ffffff; padding: 20px; text-align: center;">
                        <h2 style="margin: 0; font-size: 24px;">⚠️ Security Alert</h2>
                        <p style="margin: 5px 0 0; font-size: 14px; opacity: 0.9;">Unauthorized Access Attempt Detected</p>
                    </div>
                    <div style="padding: 30px;">
                        <p style="font-size: 16px; line-height: 1.5; margin-bottom: 20px; color: #444;">
                            The Attendance System camera has detected an <strong>unknown person</strong> in the frame. The system could not match this face to any registered student profile.
                        </p>
                        <div style="background-color: #fdf3f3; border-left: 4px solid #d32f2f; padding: 15px; margin-bottom: 20px; border-radius: 0 4px 4px 0;">
                            <p style="margin: 0; font-size: 15px;"><strong>Date & Time:</strong> {time_str}</p>
                            <p style="margin: 8px 0 0; font-size: 15px;"><strong>Location:</strong> College Campus Feed</p>
                        </div>
                        <p style="font-size: 15px; line-height: 1.5; color: #666; margin-bottom: 20px;">
                            Please review the active security camera feed immediately to verify the situation.
                        </p>
                        <div style="text-align: center; margin-top: 30px;">
                            <a href="http://localhost:3000/live-check" style="display: inline-block; background-color: #1a1a1a; color: #ffffff; text-decoration: none; padding: 12px 30px; border-radius: 6px; font-weight: 600; font-size: 15px;">View Live System</a>
                        </div>
                    </div>
                    <div style="background-color: #f9f9f9; padding: 20px; text-align: center; font-size: 12px; color: #888; border-top: 1px solid #eee;">
                        <p style="margin: 0;">This is an automated security alert generated by the Attendance Management System.</p>
                        <p style="margin: 5px 0 0;">Do not reply to this email.</p>
                    </div>
                </div>
            </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        print("[INFO] Security Alert Email sent to Admin!")
    except Exception as e:
        print(f"[ERROR] Failed to send email alert: {e}")

def trigger_unknown_person_alert():
    global last_unknown_alert_time
    current_time = time.time()
    if current_time - last_unknown_alert_time > ALERT_COOLDOWN_SECONDS:
        last_unknown_alert_time = current_time
        threading.Thread(target=send_unknown_alert_email, daemon=True).start()

def normalize_embedding(embedding):
    """
    Normalizes raw embedding output into a 1D float vector.
    Returns None for invalid/empty vectors.
    """
    try:
        if embedding is None:
            return None
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        if emb.size == 0 or not np.all(np.isfinite(emb)):
            return None
        return emb
    except Exception:
        return None

def cosine_similarity(embedding_a, embedding_b):
    """
    Safe cosine similarity that returns None when vectors are incompatible.
    """
    emb_a = normalize_embedding(embedding_a)
    emb_b = normalize_embedding(embedding_b)
    if emb_a is None or emb_b is None:
        return None
    if emb_a.size != emb_b.size:
        return None

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        return None

    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

def get_face_embedding(image, silent=False):
    """
    Detects face and returns a Histogram 'embedding' for comparison.
    Returns (embedding, error_string)
    """
    if image is None:
        if not silent: print("[ERROR] No image provided to get_face_embedding")
        return None, "No Image Provided"
        
    height, width, _ = image.shape
    # if not silent: print(f"[DEBUG] Processing image: {width}x{height}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detector = get_face_detector()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = detector.detect(mp_image)

    if not results or not results.detections:
        if not silent: print(f"[WARNING] No face detected by MediaPipe in {width}x{height} image")
        return None, "MediaPipe: Face not detected. Need better lighting."

    if not silent: print(f"[INFO] Detected {len(results.detections)} face(s)")

    detection = results.detections[0]
    bbox = detection.bounding_box
    x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
    
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(width, x + w)
    y_end = min(height, y + h)

    face_crop = image[y_start:y_end, x_start:x_end]
    if face_crop.size == 0: return None, "Cropped face is empty"

    from deepface import DeepFace
    try:
        # Get dense neural network embedding explicitly with Facenet
        embedding_obj = DeepFace.represent(img_path=face_crop, model_name="Facenet", detector_backend="skip", enforce_detection=False)[0]
        embedding = normalize_embedding(embedding_obj.get("embedding"))
        if embedding is None:
            return None, "Invalid embedding generated"
        return embedding, None
    except Exception as e:
        if not silent: print(f"[ERROR] DeepFace embedding failed: {e}")
        return None, f"DeepFace Error: {str(e)}"

def load_known_faces():
    global known_faces
    known_faces = []
    
    # 1. Load from Disk
    if os.path.exists(STUDENT_IMAGES_DIR):
        all_files = glob.glob(os.path.join(STUDENT_IMAGES_DIR, "**", "*.*"), recursive=True)
        valid_extensions = {".jpg", ".jpeg", ".png"}
        img_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_extensions]
        
        for file_path in img_files:
            img = cv2.imread(file_path)
            if img is not None:
                emb, _ = get_face_embedding(img, silent=True)
                emb = normalize_embedding(emb)
                if emb is not None:
                    name = os.path.basename(os.path.dirname(file_path)) or os.path.splitext(os.path.basename(file_path))[0]
                    # Try to associate with database ID for attendance records
                    student_doc = students_collection.find_one({"name": name})
                    student_id = str(student_doc["_id"]) if student_doc else None
                    known_faces.append({"id": student_id, "name": name, "embeddings": [emb.tolist()]})

    # 2. Load from Database
    db_students = list(students_collection.find({"$or": [{"faceEmbedding": {"$exists": True}}, {"faceEmbeddings": {"$exists": True}}]}))
    for s in db_students:
        try:
            embedding_candidates = []

            # New Multi-Sample Schema
            if isinstance(s.get("faceEmbeddings"), list):
                embedding_candidates.extend(s["faceEmbeddings"])

            # Legacy Single-Sample Schema
            if s.get("faceEmbedding") is not None:
                embedding_candidates.append(s["faceEmbedding"])

            normalized_embeddings = []
            for emb in embedding_candidates:
                emb_vec = normalize_embedding(emb)
                if emb_vec is None:
                    continue
                normalized_embeddings.append(emb_vec.tolist())

            # Compatibility fallback: derive at least one embedding from stored profile image.
            # This helps when legacy DB embeddings use a different dimension/model.
            image_candidates = []
            profile_image = s.get("profileImage")
            if isinstance(profile_image, str) and profile_image.strip():
                profile_filename = os.path.basename(profile_image)
                if profile_filename:
                    image_candidates.append(os.path.join(UPLOAD_DIR, profile_filename))
                    image_candidates.append(os.path.join("uploads", profile_filename))

            roll_no = str(s.get("rollNo", "")).strip()
            if roll_no:
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    image_candidates.extend(glob.glob(os.path.join(UPLOAD_DIR, f"*{roll_no}*{ext[1:]}")))

            for image_path in image_candidates:
                if not os.path.exists(image_path):
                    continue
                profile_img = cv2.imread(image_path)
                if profile_img is None:
                    continue
                profile_emb, _ = get_face_embedding(profile_img, silent=True)
                profile_emb = normalize_embedding(profile_emb)
                if profile_emb is not None:
                    normalized_embeddings.append(profile_emb.tolist())
                    break

            if normalized_embeddings:
                known_faces.append({
                    "id": str(s["_id"]),
                    "name": s.get("name", "Unknown"),
                    "embeddings": normalized_embeddings,
                })
        except Exception as e:
            print(f"[ERROR] Loading face for {s.get('name')}: {e}")
            continue

    embedding_dims = {}
    for person in known_faces:
        for emb in person.get("embeddings", []):
            emb_len = len(emb) if isinstance(emb, list) else 0
            embedding_dims[emb_len] = embedding_dims.get(emb_len, 0) + 1

    print(f"[INFO] Total loaded reference faces: {len(known_faces)} | Embedding dimensions: {embedding_dims}")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    print("[INFO] Cleaning up legacy database indexes...")
    try:
        # Log indexes for debugging
        idxs = students_collection.index_information()
        print(f"[INFO] Current Student Indexes: {list(idxs.keys())}")
        
        # Drop the problematic index if it exists
        if "enrollmentNumber_1" in idxs:
            students_collection.drop_index("enrollmentNumber_1")
            print("[INFO] Dropped legacy enrollmentNumber index.")
    except Exception as e:
        print(f"[DEBUG] Index cleanup note: {e}")

    print("[INFO] Starting face cache loader in background...")
    threading.Thread(target=load_known_faces, daemon=True).start()

# ==== AUTH ROUTES ====

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/admin/login")
async def login(data: LoginRequest):
    # For demo, allow hardcoded or any existing db admin
    admin_email_env = os.getenv("ADMIN_EMAIL", "admin@sinhgad.edu")
    admin_pass_env = os.getenv("ADMIN_PASSWORD", "Admin@123")
    
    if (data.email in ["admin@dtu.ac.in", "admin@vidya.com", admin_email_env]) and data.password == admin_pass_env:
        return {"user": {"name": "Administrator", "email": data.email, "role": "admin"}}
    
    admin = admin_collection.find_one({"email": data.email, "password": data.password})
    if admin:
        return {"user": {"name": admin.get("name", "Admin"), "email": admin["email"], "role": "admin"}}
    
    raise HTTPException(status_code=401, detail="Invalid Credentials")

# Student Login
class StudentLoginRequest(BaseModel):
    email: str
    rollNo: str

@app.post("/student/login")
async def student_login(data: StudentLoginRequest):
    student = students_collection.find_one({"email": data.email, "rollNo": data.rollNo})
    if not student:
        raise HTTPException(status_code=401, detail="Invalid Email or Roll Number")
    
    return {
        "user": {
            "id": str(student["_id"]),
            "name": student["name"],
            "email": student["email"],
            "rollNo": student["rollNo"],
            "department": student.get("department", "General"),
            "profileImage": student.get("profileImage", ""),
            "role": "student"
        }
    }

@app.get("/student/me/{student_id}")
async def get_student_profile(student_id: str):
    try:
        student = students_collection.find_one({"_id": ObjectId(student_id)})
        if not student:
             raise HTTPException(status_code=404, detail="Student not found")
        
        # Get Attendance Stats (Robust match for both string and ObjectId)
        query = {"$or": [{"studentId": student_id}, {"studentId": ObjectId(student_id)}]}
        attendance_records = list(attendance_collection.find(query).sort("date", -1))
        
        print(f"[DEBUG] Fetching Profile for {student_id}: Found {len(attendance_records)} records")
        
        # Calculate dynamic working days (unique attendance dates in system)
        unique_dates = attendance_collection.distinct("date")
        total_days = max(1, len(unique_dates)) 
        present_days = len(attendance_records)
        percentage = (present_days / total_days * 100)
        
        return {
            "profile": {
                "name": student["name"],
                "rollNo": student["rollNo"],
                "department": student.get("department", "General"),
                "email": student["email"],
                "phone": student.get("phone", ""),
                "profileImage": student.get("profileImage", "")
            },
            "stats": {
                "totalWorkingDays": total_days,
                "presentDays": present_days,
                "absentDays": total_days - present_days,
                "percentage": round(percentage, 1)
            },
            "history": [
                {
                    "date": r.get("date", ""),
                    "time": r.get("time", ""),
                    "entryTime": r.get("entryTime", ""),
                    "exitTime": r.get("exitTime", ""),
                    "status": "Present"
                } for r in attendance_records
            ]
        }
    except Exception as e:
         print(e)
         raise HTTPException(status_code=500, detail="Failed to fetch profile")

# ==== STUDENT ROUTES ====

@app.get("/students/")
async def get_students():
    students = list(students_collection.find())
    for s in students:
        s["id"] = str(s["_id"]) # Frontend expects 'id'
        s["_id"] = str(s["_id"])
        if "faceEmbedding" in s: del s["faceEmbedding"]
    return students

class StudentAddRequest(BaseModel):
    name: str
    rollNo: str
    department: str
    email: str
    phone: str
    images: List[str] # List of Base64 strings

@app.post("/students/add")
async def add_student(student: StudentAddRequest):
    if students_collection.find_one({"rollNo": student.rollNo}):
        raise HTTPException(status_code=400, detail="Student already exists")
    if students_collection.find_one({"email": student.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    if not student.images:
        raise HTTPException(status_code=400, detail="No images provided")

    # ---- FACE DEDUPLICATION CHECK (Disabled for compatibility) ----
    # if len(known_faces) > 0:
    #     for img_b64 in student.images[:2]: # Check first 2 samples for speed
    #         try:
    #             temp_b64 = img_b64.split(",")[1] if "," in img_b64 else img_b64
    #             temp_arr = np.frombuffer(base64.b64decode(temp_b64), np.uint8)
    #             temp_img = cv2.imdecode(temp_arr, cv2.IMREAD_COLOR)
    #             temp_emb = get_face_embedding(temp_img, silent=True)
    #             
    #             if temp_emb is not None:
    #                 for person in known_faces:
    #                     for stored_emb in person.get("embeddings", []):
    #                         stored_mat = np.array(stored_emb, dtype=np.float32).reshape((32, 32))
    #                         score = cv2.compareHist(temp_emb, stored_mat, cv2.HISTCMP_CORREL)
    #                         if score > 0.8: # Very strict match
    #                             raise HTTPException(status_code=400, detail=f"Face already registered as {person['name']}")
    #         except HTTPException as e: raise e
    #         except: continue

    embeddings = []
    saved_profile_image = ""
    
    # Ensure upload dir exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    print(f"[INFO] Processing {len(student.images)} images for {student.name}")

    for idx, img_b64 in enumerate(student.images):
        try:
            # Decode Base64
            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]
            
            image_data = base64.b64decode(img_b64)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None: continue

            # Generate Embedding
            emb, err = get_face_embedding(img, silent=False)
            if emb is not None:
                emb_vec = normalize_embedding(emb)
                if emb_vec is None:
                    continue
                embeddings.append(emb_vec.tolist())
                
                # Save first valid image as profile picture
                if not saved_profile_image:
                    filename = f"{student.rollNo}_profile.jpg"
                    filepath = os.path.join(UPLOAD_DIR, filename)
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    saved_profile_image = f"/uploads/{filename}"
            
        except Exception as e:
            print(f"[ERROR] Processing image {idx}: {e}")
            continue

    if not embeddings:
         raise HTTPException(status_code=400, detail="Could not detect face in any provided images")

    student_data = {
        "name": student.name,
        "rollNo": student.rollNo,
        "department": student.department,
        "email": student.email,
        "phone": student.phone,
        "profileImage": saved_profile_image,
        "faceEmbeddings": embeddings, # Store ARRAY of embeddings
        "createdAt": datetime.now()
    }
    
    result = students_collection.insert_one(student_data)
    load_known_faces() # Reload cache
    return {"id": str(result.inserted_id), "message": f"Student added with {len(embeddings)} face samples"}

@app.delete("/students/{id}")
async def delete_student(id: str):
    student = students_collection.find_one({"_id": ObjectId(id)})
    if student:
        # Delete Profile Image from disk
        profile_img = student.get("profileImage")
        if profile_img:
            filename = os.path.basename(profile_img)
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(filepath):
                try: os.remove(filepath)
                except: pass
    
    students_collection.delete_one({"_id": ObjectId(id)})
    load_known_faces() # Reload cache to remove deleted student
    return {"message": "Deleted"}

class StudentUpdateRequest(BaseModel):
    name: str = None
    rollNo: str = None
    department: str = None
    email: str = None
    phone: str = None

@app.put("/students/{id}")
async def update_student(id: str, student: StudentUpdateRequest):
    update_data = {k: v for k, v in student.dict().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Check for duplicate rollNo if changing it
    if "rollNo" in update_data:
        existing = students_collection.find_one({"rollNo": update_data["rollNo"]})
        if existing and str(existing["_id"]) != id:
             raise HTTPException(status_code=400, detail="Roll No already exists")

    result = students_collection.update_one(
        {"_id": ObjectId(id)}, 
        {"$set": update_data}
    )
    
    if result.modified_count == 0:
         raise HTTPException(status_code=404, detail="Student not found or no changes made")
         
    load_known_faces() # Reload cache to update changes (e.g. name)
    return {"message": "Student updated successfully"}

# ==== MODELS ====

class AttendanceRequest(BaseModel):
    image: str

# ==== FACE RECOGNITION (LIVE CHECK) ROUTE ====

@app.post("/face/recognize")
async def recognize_face(data: AttendanceRequest):
    """
    Stand-alone recognition endpoint for 'Live Check' page.
    Does NOT mark attendance.
    """
    if not data.image:
        raise HTTPException(status_code=400, detail="No image")

    try:
        if len(known_faces) == 0:
            return {"status": "fail", "message": "No registered student faces loaded"}

        encoded = data.image.split(",", 1)[1] if "," in data.image else data.image
        np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None: raise HTTPException(status_code=400, detail="Invalid Image")

        target_emb, err_msg = get_face_embedding(img, silent=True)
        if target_emb is None:
             return {"status": "fail", "message": err_msg}

        best_score = 0
        best_match = None
        comparison_count = 0
        
        for person in known_faces:
            local_best = 0
            for emb_list in person.get("embeddings", []):
                score = cosine_similarity(target_emb, emb_list)
                if score is None:
                    continue
                comparison_count += 1
                if score > local_best: local_best = score
            
            if local_best > best_score:
                best_score = local_best
                best_match = person

        if comparison_count == 0:
            return {"status": "fail", "message": "Stored embeddings are incompatible. Re-register face samples."}

        if best_match and best_score > SIMILARITY_THRESHOLD:
            # Fetch full details to populate Live Check UI
            student_doc = {}
            if best_match.get("id"):
                student_doc = students_collection.find_one({"_id": ObjectId(best_match["id"])}) or {}
                
            return {
                "status": "success",
                "student": {
                    "name": best_match["name"],
                    "id": best_match["id"],
                    "email": student_doc.get("email", ""),
                    "phone": student_doc.get("phone", ""),
                    "department": student_doc.get("department", "General"),
                    "profileImage": student_doc.get("profileImage", "")
                },
                "score": float(round(best_score, 2))
            }
        
        trigger_unknown_person_alert()
        return {"status": "fail", "message": "Unknown Student", "score": float(round(best_score, 2))}

    except Exception as e:
        print(f"Recognition Error: {e}")
        return {"status": "error", "message": "Server Error"}


@app.post("/attendance/mark")
async def mark_attendance(data: AttendanceRequest):
    if not data.image:
        raise HTTPException(status_code=400, detail="No image")

    try:
        encoded = data.image.split(",", 1)[1] if "," in data.image else data.image
        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None: raise HTTPException(status_code=400, detail="Invalid Image")

        # Use High-Quality Detection for Verification
        detector_hq = get_face_detector()
        mp_image_hq = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results = detector_hq.detect(mp_image_hq)
        
        if not results.detections:
             raise HTTPException(status_code=400, detail="No face detected. Please position better.")
        
        # Get target embedding
        # We need a robust embedding. Re-using the utility function but ensuring it uses the cropped face
        target_emb, _ = get_face_embedding(img, silent=True)
        if target_emb is None:
             raise HTTPException(status_code=400, detail="Face quality too low")

        best_score = 0
        best_match_id = None
        comparison_count = 0
        
        # known_faces stores: { "id": str(_id), "embeddings": [list...], "name": ... }
        for person in known_faces:
            # We compare against all stored embeddings for this person
            person_embeddings = person.get("embeddings", [])
            
            # New Multi-Embedding Check (Best of Max) using Cosine Similarity
            local_best = 0
            for emb_list in person_embeddings:
                score = cosine_similarity(target_emb, emb_list)
                if score is None:
                    continue
                comparison_count += 1
                if score > local_best: local_best = score
            
            if local_best > best_score:
                best_score = local_best
                best_match_id = person.get("id")

        if comparison_count == 0:
            raise HTTPException(status_code=500, detail="Stored embeddings are incompatible. Re-register student face samples.")

        # detector_hq.close() not needed with singleton
        print(f"[FACE AUTH] Best Match ID: {best_match_id} | Score: {best_score:.4f} | Threshold: {SIMILARITY_THRESHOLD}")

        if best_match_id and best_score > SIMILARITY_THRESHOLD:
            # Fetch Student Details
            student = students_collection.find_one({"_id": ObjectId(best_match_id)})
            if not student:
                print(f"[ERROR] Matched ID {best_match_id} but not in DB")
                raise HTTPException(status_code=404, detail="Student record not found")
        else:
             print(f"[AUTH FAIL] Best Score: {best_score} vs Threshold {SIMILARITY_THRESHOLD}")
             msg = f"Face Not Recognized. Score: {best_score:.2f} (Needs {SIMILARITY_THRESHOLD}). Try better lighting."
             if len(known_faces) == 0:
                 msg = "System Error: No student faces loaded in database. Restart Backend."
             else:
                 trigger_unknown_person_alert()
             raise HTTPException(status_code=401, detail=msg)

        today = datetime.now().strftime("%Y-%m-%d")
        # Robust check for existing record (string or ObjectId)
        existing_query = {
            "$or": [{"studentId": str(student["_id"])}, {"studentId": student["_id"]}], 
            "date": today
        }
        existing = attendance_collection.find_one(existing_query)
        print(f"[DEBUG] Check Existing for {student['name']}: {'Found' if existing else 'Not Found'}")
        
        current_time_str = datetime.now().strftime("%H:%M:%S")

        if not existing:
            new_record = {
                "studentId": str(student["_id"]), 
                "studentName": student["name"],
                "rollNo": student["rollNo"],
                "date": today,
                "entryTime": current_time_str,
                "exitTime": None,
                "time": current_time_str,
                "status": "Present"
            }
            res = attendance_collection.insert_one(new_record)
            print(f"[DEBUG] Inserted ENTRY record for {student['name']}. ID: {res.inserted_id}")
            return {
                "status": "success", 
                "message": f"Entry Marked: {student['name']} at {current_time_str}",
                "student": {"name": student["name"], "rollNo": student["rollNo"]}
            }
        else:
            attendance_collection.update_one(
                {"_id": existing["_id"]},
                {"$set": {"exitTime": current_time_str}}
            )
            print(f"[DEBUG] Updated EXIT record for {student['name']} to {current_time_str}.")
            return {
                "status": "success", 
                "message": f"Exit Marked: {student['name']} at {current_time_str}",
                "student": {"name": student["name"], "rollNo": student["rollNo"]}
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error marking attendance: {e}")
        raise HTTPException(status_code=500, detail="Server Error Processing Image")

@app.get("/attendance/today")
async def get_today():
    today = datetime.now().strftime("%Y-%m-%d")
    records = list(attendance_collection.find({"date": today}))
    print(f"[DEBUG] Today's Attendance Request: Found {len(records)} records for {today}")
    
    # Manual serialization to ensure no ObjectId leaks
    cleaned_records = []
    for r in records:
        r["_id"] = str(r["_id"])
        if "studentId" in r and isinstance(r["studentId"], ObjectId):
             r["studentId"] = str(r["studentId"])
        cleaned_records.append(r)
        
    return cleaned_records

@app.get("/attendance/stats")
async def get_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    total = students_collection.count_documents({})
    present = attendance_collection.count_documents({"date": today})
    print(f"[DEBUG] Stats Request - Today: {today} | Total: {total} | Present: {present}")
    return {
        "totalStudents": total,
        "presentToday": present,
        "absentToday": max(0, total - present),
        "attendancePercentage": round((present / total * 100), 1) if total > 0 else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
