import logging
from src.WhatsApp_Analyser.utils.abstract import pipeline
from utils.asyncHandler import asyncHandler
from src.WhatsApp_Analyser.components.DataIngestion_component import DataIngestionComponent
from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact
from src.WhatsApp_Analyser.entity.config_entity import DataIngestionConfig

logger = logging.getLogger(__name__)

class DataIngestionPipeline(pipeline):
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        logger.info("Initializing DataIngestionPipeline")
        self.data_ingestion_config=data_ingestion_config

    @asyncHandler   
    async def initiate(self)->DataIngestionArtifact:
        logger.info("Starting Data Ingestion Pipeline")
        data_ingestion=DataIngestionComponent(data_ingestion_config=self.data_ingestion_config)
        artifact = await data_ingestion.ingest()
        logger.info("Data Ingestion Pipeline completed successfully")
        return artifact
