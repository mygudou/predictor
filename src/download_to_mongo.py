import yfinance as yf
import pandas as pd
from pymongo import MongoClient

class ShanghaiCompositeDownloader:
    def __init__(self, db_name="finance", collection_name="shanghai_composite", uri="mongodb://localhost:27017/"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def download_data(self, start_date="2000-01-01", end_date=None):
        """
        使用 yfinance 下载上证指数历史数据。
        """
        print("Downloading Shanghai Composite Index data...")
        ticker = "000001.SS"  # 上证指数在 Yahoo Finance 中的代码
        data = yf.download(ticker, start=start_date, end=end_date)

        # Reset the index to make the Date column accessible
        data.reset_index(inplace=True)

        # Flatten and clean column names
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # Rename columns to remove any ambiguity
        data.rename(columns={"Date": "Date", "Close": "Close"}, inplace=True)
        print("Data downloaded successfully.")
        print(data)
        return data

    def save_to_mongo(self, data: pd.DataFrame):
        """
        将数据保存到 MongoDB。
        """
        print("Saving data to MongoDB...")
        self.collection.delete_many({})  # 清空旧数据

        # 确保所有列名是字符串，转换日期为字符串格式
        data['Date'] = data['Date'].astype(str)  # 将日期列转换为字符串
        records = data.to_dict("records")

        self.collection.insert_many(records)
        print("Data saved successfully.")


if __name__ == "__main__":
    downloader = ShanghaiCompositeDownloader()
    # 下载上证指数数据
    historical_data = downloader.download_data(start_date="2000-01-01")
    # 保存到 MongoDB
    downloader.save_to_mongo(historical_data)
