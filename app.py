import sys
import os
import certifi
import io
import shutil
import glob
import pandas as pd
import pymongo
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

# Project core imports
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.constants.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

# Configuration & DB Setup
load_dotenv()
ca = certifi.where()
mongo_db_url = os.getenv("MONGODB_URL_KEY")

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory="./templates")

# ==========================================
# ABSOLUTE PATH CONFIGURATION (ROOT LEVEL)
# ==========================================
# Resolves to your main project folder root
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# If APP_DIR points to 'venv', step out to project root
if os.path.basename(APP_DIR) == "venv":
    APP_DIR = os.path.dirname(APP_DIR)

FINAL_MODEL_DIR = os.path.join(APP_DIR, "final_model")
PREPROCESSOR_PATH = os.path.join(FINAL_MODEL_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(FINAL_MODEL_DIR, "model.pkl")

# Safely ensure the final_model folder exists on disk
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# Global runtime object
network_model = None

def load_global_model():
    """Helper function to load model and preprocessor into runtime memory"""
    global network_model
    try:
        if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
            global_preprocessor = load_object(PREPROCESSOR_PATH)
            global_final_model = load_object(MODEL_PATH)
            network_model = NetworkModel(preprocessor=global_preprocessor, model=global_final_model)
            logging.info("Model and preprocessor loaded successfully into runtime memory.")
        else:
            network_model = None
            logging.warning("Model files not found in final_model folder yet.")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

# Initial try-load when server starts up
load_global_model()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train") 
async def train_route():
    try:
        # 1. Run the ML Pipeline
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
        # 2. Automatically locate the newly trained pipeline files
        artifact_dir = os.path.join(APP_DIR, "artifact")
        
        # Find the freshest model and preprocessor files inside the artifacts folder tree
        latest_preprocessors = glob.glob(os.path.join(artifact_dir, "**", "transformed", "preprocessor.pkl"), recursive=True) or \
                               glob.glob(os.path.join(artifact_dir, "**", "preprocessor.pkl"), recursive=True)
        latest_models = glob.glob(os.path.join(artifact_dir, "**", "model_trainer", "model.pkl"), recursive=True) or \
                        glob.glob(os.path.join(artifact_dir, "**", "model.pkl"), recursive=True)
        
        if latest_preprocessors and latest_models:
            # Sort by modification time to get the newest run files
            latest_preprocessors.sort(key=os.path.getmtime)
            latest_models.sort(key=os.path.getmtime)
            
            # Copy artifacts directly into your production 'final_model' tracking folder
            shutil.copy(latest_preprocessors[-1], PREPROCESSOR_PATH)
            shutil.copy(latest_models[-1], MODEL_PATH)
            logging.info("Successfully synchronized fresh pipeline artifacts to final_model directory.")
        
        # 3. Hot-reload the memory runtime object
        load_global_model()
        
        return Response("Training successful and artifacts updated.")
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        global network_model
        if network_model is None:
            # Fallback check in case files were manually pasted while server was running
            load_global_model()
            if network_model is None:
                raise Exception("Machine learning model is not initialized. Please hit the /train endpoint first.")
            
        # Read file stream safely into memory
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Inference using the pre-loaded architecture
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred
        
        # Ensure target directories exist before writing output files
        os.makedirs(os.path.join(APP_DIR, "predict_output"), exist_ok=True)
        df.to_csv(os.path.join(APP_DIR, "predict_output", "output.csv"), index=False)
        
        table_html = df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
            
    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)