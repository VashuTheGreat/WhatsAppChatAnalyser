import logging
import logger
from src.WhatsApp_Analyser.utils.abstract import pipeline
from utils.asyncHandler import asyncHandler
from src.WhatsApp_Analyser.components.DataTransformation_config import DataTransformationComponent
from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.WhatsApp_Analyser.entity.config_entity import DataTransformationConfig

logger_obj = logging.getLogger(__name__)

class DataTransformationPipeline(pipeline):
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        logger_obj.info("Initializing DataTransformationPipeline")
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    @asyncHandler
    async def initiate(self) -> DataTransformationArtifact:
        logger_obj.info("Starting Data Transformation Pipeline")
        data_transformation = DataTransformationComponent(
            data_transformation_config=self.data_transformation_config,
            data_ingestion_artifact=self.data_ingestion_artifact
        )
        artifact = await data_transformation.transform()
        logger_obj.info("Data Transformation Pipeline completed successfully")
        return artifact
