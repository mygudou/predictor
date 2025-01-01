from pymongo import MongoClient
import pandas as pd


class MongoDBHandler:
    def __init__(self, db_name="finance", collection_name="shanghai_composite", uri="mongodb://localhost:27017/"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_data(self, data: pd.DataFrame):
        # 插入数据到 MongoDB
        records = data.to_dict('records')
        self.collection.insert_many(records)

    def fetch_data(self):
        # 从 MongoDB 中加载数据为 DataFrame
        cursor = self.collection.find({}, {"_id": 0})  # 不返回 _id 字段
        return pd.DataFrame(list(cursor))
