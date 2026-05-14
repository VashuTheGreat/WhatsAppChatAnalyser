import logging
import logger
from src.WhatsApp_Analyser.utils.abstract import pipeline
from utils.asyncHandler import asyncHandler
from src.WhatsApp_Analyser.components.DataValidation_compoent import DataValidationComponent
from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.WhatsApp_Analyser.entity.config_entity import DataValidationConfig

logger_obj = logging.getLogger(__name__)

class DataValidationPipeline(pipeline):
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        logger_obj.info("Initializing DataValidationPipeline")
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    @asyncHandler
    async def initiate(self) -> DataValidationArtifact:
        logger_obj.info("Starting Data Validation Pipeline")
        data_validation = DataValidationComponent(
            data_validation_config=self.data_validation_config,
            data_ingestion_artifact=self.data_ingestion_artifact
        )
        artifact = await data_validation.validate()
        logger_obj.info("Data Validation Pipeline completed successfully")
        return artifact
