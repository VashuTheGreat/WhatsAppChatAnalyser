# import pytest

# import sys
# import os
# sys.path.append(os.getcwd())
# import logger

# import logging
# import matplotlib.pyplot as plt
# from src.WhatsApp_Analyser.pipelines.data_ingestion_pipeline import DataIngestionPipeline
# from src.WhatsApp_Analyser.pipelines.data_validation_pipeline import DataValidationPipeline
# from src.WhatsApp_Analyser.pipelines.data_transformation_pipeline import DataTransformationPipeline
# from src.WhatsApp_Analyser.pipelines.data_analyser_pipeline import DataAnalyserPipeline
# from src.WhatsApp_Analyser.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, DataAnalyserConfig
# from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, DataAnalyserArtifact

# @pytest.mark.asyncio
# async def test_data_analyser_pipeline():
#     logging.info("Starting data analyser pipeline test")
    
#     data_path = "data/WhatsApp Chat with CSE AIML-2 Unofficial.txt"
#     data_ingestion_config = DataIngestionConfig(ingest_file_path=data_path)
#     data_ingestion_pipeline = DataIngestionPipeline(data_ingestion_config=data_ingestion_config)
#     data_ingestion_artifact = await data_ingestion_pipeline.initiate()
    
#     data_validation_config = DataValidationConfig()
#     data_validation_pipeline = DataValidationPipeline(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
#     data_validation_artifact = await data_validation_pipeline.initiate()
#     assert data_validation_artifact.validation is True
    
#     data_transformation_config = DataTransformationConfig()
#     data_transformation_pipeline = DataTransformationPipeline(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
#     data_transformation_artifact = await data_transformation_pipeline.initiate()
    
#     data_analyser_config = DataAnalyserConfig()
#     data_analyser_pipeline = DataAnalyserPipeline(data_analyser_config=data_analyser_config, data_transformation_artifact=data_transformation_artifact)
#     data_analyser_artifact = await data_analyser_pipeline.initiate()
    
#     assert isinstance(data_analyser_artifact, DataAnalyserArtifact)
#     report = data_analyser_artifact.analysis_report
#     assert "stats" in report
#     assert "plots" in report
#     assert isinstance(report["plots"]["day_timeline"], plt.Figure)
