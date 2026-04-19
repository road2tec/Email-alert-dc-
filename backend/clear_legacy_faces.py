from pymongo import MongoClient

def reset_db():
    client = MongoClient("mongodb://127.0.0.1:27017/")
    db = client["attendance"] # The collection is established in app.py via mongodb://localhost:27017/attendance
    
    res1 = db["students"].delete_many({})
    res2 = db["attendance"].delete_many({})
    
    print(f"Deleted {res1.deleted_count} students (legacy embeddings stripped).")
    print(f"Deleted {res2.deleted_count} attendance records.")

if __name__ == '__main__':
    reset_db()
