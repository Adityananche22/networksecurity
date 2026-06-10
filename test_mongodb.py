from pymongo import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://adityananche22:<@password>@cluster0.v9iac3j.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)