from pymongo import MongoClient
import json
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["vidya-rakshak"]
att = list(db["attendance"].find())
for a in att:
    print(f"record doc: _id={a.get('_id')}, studentName={a.get('studentName')}, studentId={a.get('studentId')} (type: {type(a.get('studentId'))}), date={a.get('date')}, entryTime={a.get('entryTime', None)}, exitTime={a.get('exitTime', None)}, time={a.get('time', None)}")
