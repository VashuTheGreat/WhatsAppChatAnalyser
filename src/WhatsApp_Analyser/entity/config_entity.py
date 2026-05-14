from dataclasses import dataclass

from src.WhatsApp_Analyser.constants import *
import os
from datetime import datetime
# TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
TIMESTAMP: str = ""

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config:TrainingPipelineConfig=TrainingPipelineConfig()
@dataclass
class DataIngestionConfig:
    ingest_file_path:str="None"
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    # collection_name:str = DATA_INGESTION_COLLECTION_NAME    

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)
    schema_file_path: str = DATA_VALIDATION_SCHEMA_FILE_PATH

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR, DATA_TRANSFORMATION_TRAIN_FILE_NAME)
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR, DATA_TRANSFORMATION_TEST_FILE_NAME)

@dataclass
class DataAnalyserConfig:
    data_analyser_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_ANALYSER_DIR_NAME)

