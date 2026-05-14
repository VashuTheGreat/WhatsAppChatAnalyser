import logging
import logger
import os
import pandas as pd
from src.WhatsApp_Analyser.entity.config_entity import DataTransformationConfig
from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.WhatsApp_Analyser.utils.main_utils import write_file
from utils.asyncHandler import asyncHandler

logger_obj = logging.getLogger(__name__)

class DataTransformationComponent:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        logger_obj.info("Initializing DataTransformationComponent")
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        logger_obj.info("Cleaning dataframe")
        df.dropna(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
        df['Day'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.month_name()
        df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M %p').dt.hour
        return df

    @asyncHandler
    async def transform(self) -> DataTransformationArtifact:
        logger_obj.info("Starting data transformation")
        
        train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
        test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
        
        logger_obj.info("Applying cleaning to train and test datasets")
        train_df = self.cleaning(train_df)
        test_df = self.cleaning(test_df)
        
        os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
        
        await write_file(self.data_transformation_config.transformed_train_file_path, train_df)
        logger_obj.info(f"Transformed train data saved at {self.data_transformation_config.transformed_train_file_path}")
        
        await write_file(self.data_transformation_config.transformed_test_file_path, test_df)
        logger_obj.info(f"Transformed test data saved at {self.data_transformation_config.transformed_test_file_path}")
        
        artifact = DataTransformationArtifact(
            transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
        )
        
        logger_obj.info("Data transformation completed successfully")
        return artifact
