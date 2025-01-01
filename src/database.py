from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

class MongoDBHandler:
    def __init__(self):
        # 从环境变量中读取数据库配置
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("DB_NAME", "finance")
        collection_name = os.getenv("COLLECTION_NAME", "shanghai_composite")

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
