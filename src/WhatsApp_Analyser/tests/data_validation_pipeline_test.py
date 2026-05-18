# import pytest
# import logging
# import logger
# from src.WhatsApp_Analyser.pipelines.data_ingestion_pipeline import DataIngestionPipeline
# from src.WhatsApp_Analyser.pipelines.data_validation_pipeline import DataValidationPipeline
# from src.WhatsApp_Analyser.entity.config_entity import DataIngestionConfig, DataValidationConfig
# from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

# @pytest.mark.asyncio
# async def test_data_validation_pipeline():
#     logging.info("Starting data validation pipeline test")
    
#     data_path = "data/WhatsApp Chat with CSE AIML-2 Unofficial.txt"
#     data_ingestion_config = DataIngestionConfig(ingest_file_path=data_path)
#     data_ingestion_pipeline = DataIngestionPipeline(data_ingestion_config=data_ingestion_config)
#     data_ingestion_artifact = await data_ingestion_pipeline.initiate()
    
#     assert isinstance(data_ingestion_artifact, DataIngestionArtifact)
    
#     data_validation_config = DataValidationConfig()
#     data_validation_pipeline = DataValidationPipeline(
#         data_validation_config=data_validation_config,
#         data_ingestion_artifact=data_ingestion_artifact
#     )
    
#     data_validation_artifact = await data_validation_pipeline.initiate()
    
#     logging.info(f"Data validation artifact: {data_validation_artifact}")
    
#     assert isinstance(data_validation_artifact, DataValidationArtifact)
#     assert data_validation_artifact.validation is True
#     assert data_validation_artifact.message == "Validation Success"
