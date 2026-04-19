from pymongo import MongoClient
import json
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["vidya-rakshak"]
students = list(db["students"].find())
print("Total DB Students:", len(students))
for s in students:
    if "faceEmbeddings" in s:
        print(f"Name: {s.get('name')}, Embeddings num: {len(s['faceEmbeddings'])}, Length of first elem: {len(s['faceEmbeddings'][0]) if s['faceEmbeddings'] else 0}")
    elif "faceEmbedding" in s:
        print(f"Name: {s.get('name')}, Single Embedding len: {len(s['faceEmbedding'])}")
    else:
        print(f"Name: {s.get('name')}, NO EMBEDDINGS")
