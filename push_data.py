import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

import certifi
import pandas as pd
import numpy as np
import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

ca = certifi.where()


class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)

            # Replace NaN values with None
            data = data.replace({np.nan: None})

            data.reset_index(drop=True, inplace=True)

            records = list(json.loads(data.T.to_json()).values())

            return records

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            # Connect to MongoDB
            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=ca
            )

            # Select database
            db = self.mongo_client[database]

            # Select collection
            collection_obj = db[collection]

            # Insert records
            result = collection_obj.insert_many(records)

            logging.info(
                f"{len(result.inserted_ids)} records inserted successfully."
            )

            return len(result.inserted_ids)

        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":

    FILE_PATH = r"E:\Aditya\course\udemy final\NETWORKSECURITY\Network_data\phisingData.csv"

    DATABASE = "ADITYAAI"
    COLLECTION = "NetworkData"

    network_obj = NetworkDataExtract()

    print("Converting CSV to JSON records...")

    records = network_obj.csv_to_json_converter(
        file_path=FILE_PATH
    )

    print(f"Total Records Found: {len(records)}")

    print("Pushing data to MongoDB...")

    no_of_records = network_obj.insert_data_mongodb(
        records=records,
        database=DATABASE,
        collection=COLLECTION
    )

    print(f"Successfully inserted {no_of_records} records into MongoDB.")