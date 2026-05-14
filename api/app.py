import io
import base64
import logging
import os
from typing import List, Tuple, Any

import nltk
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.WhatsApp_Analyser.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.WhatsApp_Analyser.pipelines.data_validation_pipeline import DataValidationPipeline
from src.WhatsApp_Analyser.pipelines.data_transformation_pipeline import DataTransformationPipeline
from src.WhatsApp_Analyser.pipelines.data_analyser_pipeline import DataAnalyserPipeline
from src.WhatsApp_Analyser.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, DataAnalyserConfig
from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, DataAnalyserArtifact

for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

app = FastAPI()
templates = Jinja2Templates(directory="api/templates")

def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    filename_path = "temp_filename.txt"
    current_filename = None
    if os.path.exists(filename_path):
        with open(filename_path, "r") as f:
            current_filename = f.read()
    
    return templates.TemplateResponse(
        request=request, 
        name="index.html",
        context={"filename": current_filename}
    )

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request, 
    chat_file: UploadFile = File(None),
    selected_contact: str = Form(None)
):
    temp_chat_path = "temp_chat.txt"
    filename_path = "temp_filename.txt"
    
    if chat_file and chat_file.filename:
        content = await chat_file.read()
        if content:
            with open(temp_chat_path, "wb") as f:
                f.write(content)
            with open(filename_path, "w") as f:
                f.write(chat_file.filename)
    
    current_filename = "Uploaded File"
    if os.path.exists(filename_path):
        with open(filename_path, "r") as f:
            current_filename = f.read()

    if not os.path.exists(temp_chat_path) or os.path.getsize(temp_chat_path) == 0:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "Please upload a chat file.",
                "filename": current_filename
            }
        )
        
    data_ingestion_config = DataIngestionConfig(ingest_file_path=temp_chat_path)
    data_ingestion_pipeline = DataIngestionPipeline(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = await data_ingestion_pipeline.initiate()
    
    data_validation_config = DataValidationConfig()
    data_validation_pipeline = DataValidationPipeline(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
    data_validation_artifact = await data_validation_pipeline.initiate()
    
    if not data_validation_artifact.validation:
        return templates.TemplateResponse(
            request=request,
            name="index.html", 
            context={
                "error": f"Validation failed: {data_validation_artifact.message}",
                "filename": current_filename
            }
        )
        
    data_transformation_config = DataTransformationConfig()
    data_transformation_pipeline = DataTransformationPipeline(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
    data_transformation_artifact = await data_transformation_pipeline.initiate()
    
    data_analyser_config = DataAnalyserConfig()
    data_analyser_pipeline = DataAnalyserPipeline(data_analyser_config=data_analyser_config, data_transformation_artifact=data_transformation_artifact)
    data_analyser_artifact = await data_analyser_pipeline.initiate(selected_contact=selected_contact)
    
    report = data_analyser_artifact.analysis_report
    
    plots_base64 = {k: fig_to_base64(v) for k, v in report["plots"].items()}

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "filename": current_filename,
            "contacts": report["contacts"],
            "selected_contact": report["selected_contact"],
            "analysis_done": True,
            "stats": report["stats"],
            "plots": plots_base64,
            "busy_users_table": report["busy_users_table"].to_dict('records'),
            "emoji_table": report["emoji_table"].head(20).to_dict('records')
        }
    )

