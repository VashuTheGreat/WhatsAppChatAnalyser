import pytest
import logging
import logger
import pandas as pd
from src.WhatsApp_Analyser.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.WhatsApp_Analyser.pipelines.data_validation_pipeline import DataValidationPipeline
from src.WhatsApp_Analyser.pipelines.data_transformation_pipeline import DataTransformationPipeline
from src.WhatsApp_Analyser.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact

@pytest.mark.asyncio
async def test_data_transformation_pipeline():
    logging.info("Starting data transformation pipeline test with validation gate")
    
    data_path = "/home/vashuthegreat/Projects/WhatsAppChatAnalyser/data/WhatsApp Chat with CSE AIML-2 Unofficial.txt"
    data_ingestion_config = DataIngestionConfig(ingest_file_path=data_path)
    data_ingestion_pipeline = DataIngestionPipeline(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = await data_ingestion_pipeline.initiate()
    
    assert isinstance(data_ingestion_artifact, DataIngestionArtifact)
    
    data_validation_config = DataValidationConfig()
    data_validation_pipeline = DataValidationPipeline(
        data_validation_config=data_validation_config,
        data_ingestion_artifact=data_ingestion_artifact
    )
    data_validation_artifact = await data_validation_pipeline.initiate()
    
    assert isinstance(data_validation_artifact, DataValidationArtifact)
    
    if not data_validation_artifact.validation:
        logging.error(f"Data validation failed: {data_validation_artifact.message}")
        pytest.fail(f"Data transformation skipped due to validation failure: {data_validation_artifact.message}")
    
    data_transformation_config = DataTransformationConfig()
    data_transformation_pipeline = DataTransformationPipeline(
        data_transformation_config=data_transformation_config,
        data_ingestion_artifact=data_ingestion_artifact
    )
    
    data_transformation_artifact = await data_transformation_pipeline.initiate()
    
    logging.info(f"Data transformation artifact: {data_transformation_artifact}")
    
    assert isinstance(data_transformation_artifact, DataTransformationArtifact)
    assert data_transformation_artifact.transformed_train_file_path is not None
    assert data_transformation_artifact.transformed_test_file_path is not None
    
    train_df = pd.read_csv(data_transformation_artifact.transformed_train_file_path)
    assert 'Day' in train_df.columns
    assert 'Month' in train_df.columns
    assert 'Hour' in train_df.columns
