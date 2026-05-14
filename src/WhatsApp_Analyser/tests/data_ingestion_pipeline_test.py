import pytest
import sys
import os

from logger import *
sys.path.append(os.getcwd())
import logging
from src.WhatsApp_Analyser.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.WhatsApp_Analyser.entity.config_entity import DataIngestionConfig
from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact

@pytest.mark.asyncio
async def test_data_ingestion_pipeline():
    data_path:str="/home/vashuthegreat/Projects/WhatsAppChatAnalyser/data/WhatsApp Chat with CSE AIML-2 Unofficial.txt"
    data_ingestion_config = DataIngestionConfig(
        ingest_file_path=data_path
        
    )
    print(data_ingestion_config)
    logging.info("Running data ingestion pipeline test")
    data_ingestion_pipeline = DataIngestionPipeline(data_ingestion_config=data_ingestion_config)

    data_ingestion_artifact = await data_ingestion_pipeline.initiate()

    logging.debug(f"Got data_ingestion_artifact: {data_ingestion_artifact}")
    
    assert isinstance(data_ingestion_artifact, DataIngestionArtifact)
    assert data_ingestion_artifact.trained_file_path is not None
    assert data_ingestion_artifact.test_file_path is not None
