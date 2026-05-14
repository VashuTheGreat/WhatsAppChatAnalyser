import logging
import logger
import os
import pandas as pd
from src.WhatsApp_Analyser.entity.config_entity import DataValidationConfig
from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.WhatsApp_Analyser.utils.main_utils import load_yml, write_yml
from utils.asyncHandler import asyncHandler

logger_obj = logging.getLogger(__name__)

class DataValidationComponent:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        logger_obj.info("Initializing DataValidationComponent")
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    @asyncHandler
    async def _validate(self, schema: dict, data: pd.DataFrame) -> tuple[bool, str]:
        logger_obj.info("Starting schema validation")
        
        expected_columns = schema.get("columns", {}).keys()
        expected_num_columns = schema.get("num_columns")
        
        logger_obj.debug(f"Expected columns: {list(expected_columns)}")
        logger_obj.debug(f"Expected number of columns: {expected_num_columns}")

        if len(data.columns) != expected_num_columns:
            msg = f"Column count mismatch. Expected {expected_num_columns}, got {len(data.columns)}"
            logger_obj.error(msg)
            return False, msg

        for col in expected_columns:
            if col not in data.columns:
                msg = f"Missing column: {col}"
                logger_obj.error(msg)
                return False, msg
        
        logger_obj.info("Schema validation successful")
        return True, "Validation Success"

    @asyncHandler
    async def validate(self) -> DataValidationArtifact:
        logger_obj.info("Starting data validation process")
        
        try:
            schema = await load_yml(self.data_validation_config.schema_file_path)
            logger_obj.info(f"Schema loaded from {self.data_validation_config.schema_file_path}")

            feature_store_path = os.path.join(os.path.dirname(self.data_ingestion_artifact.trained_file_path).replace("ingested_data", "features"), "data.csv")
            data = pd.read_csv(feature_store_path)
            logger_obj.info(f"Data loaded from {feature_store_path}")

            validation, message = await self._validate(schema, data)

            data_validation_artifact = DataValidationArtifact(
                validation=validation,
                message=message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            report = {
                "validation": validation,
                "message": message
            }
            
            await write_yml(self.data_validation_config.validation_report_file_path, report)
            logger_obj.info(f"Validation report saved at {self.data_validation_config.validation_report_file_path}")

            return data_validation_artifact

        except Exception as e:
            logger_obj.exception("Error during data validation")
            return DataValidationArtifact(
                validation=False,
                message=str(e),
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )
