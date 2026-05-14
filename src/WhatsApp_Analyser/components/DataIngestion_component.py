import logging

from utils.asyncHandler import asyncHandler
from src.WhatsApp_Analyser.entity.config_entity import DataIngestionConfig
from src.WhatsApp_Analyser.entity.artifact_entity import DataIngestionArtifact
from src.WhatsApp_Analyser.data_access import DataAccess
from sklearn.model_selection import train_test_split
from src.WhatsApp_Analyser.utils.main_utils import write_file
import os
import pandas as pd

logger = logging.getLogger(__name__)

class DataIngestionComponent:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        logger.info("Initializing DataIngestionComponent")
        self.data_ingestion_config=data_ingestion_config
        self.data_access=DataAccess(url=self.data_ingestion_config.ingest_file_path)
        logger.debug(f"Config: {self.data_ingestion_config}")
    
    @asyncHandler
    async def _split(self,data:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
        logger.info("Splitting dataset into train and test")
        train,test=train_test_split(data,test_size=self.data_ingestion_config.train_test_split_ratio,shuffle=True)
        logger.debug(f"Train size: {len(train)}, Test size: {len(test)}")
        return train,test

    @asyncHandler
    async def ingest(self)->DataIngestionArtifact:
        logger.info("Starting data ingestion process")
        data:pd.DataFrame=await self.data_access.get_data()
        
        if len(data) == 0:
            raise Exception("Fetched dataset is empty. Please check the chat file format.")

        logger.info(f"Data fetched successfully. Row count: {len(data)}")
        
        train,test=await self._split(data)

        logger.info("Saving feature store, training, and testing files")

        directory_path=os.path.dirname(self.data_ingestion_config.feature_store_file_path)
        os.makedirs(directory_path,exist_ok=True)
        logger.debug(f"Feature store directory created: {directory_path}")

        await write_file(self.data_ingestion_config.feature_store_file_path,data)
        logger.info(f"Feature store file saved at: {self.data_ingestion_config.feature_store_file_path}")

        directory_path=os.path.dirname(self.data_ingestion_config.testing_file_path)
        os.makedirs(directory_path,exist_ok=True)
        logger.debug(f"Testing directory created: {directory_path}")

        await write_file(self.data_ingestion_config.testing_file_path,test)
        logger.info(f"Testing file saved at: {self.data_ingestion_config.testing_file_path}")

        directory_path=os.path.dirname(self.data_ingestion_config.training_file_path)
        os.makedirs(directory_path,exist_ok=True)
        logger.debug(f"Training directory created: {directory_path}")

        await write_file(self.data_ingestion_config.training_file_path,train)
        logger.info(f"Training file saved at: {self.data_ingestion_config.training_file_path}")

        data_ingestion_artifact:DataIngestionArtifact=DataIngestionArtifact(
            trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path
        )

        logger.info("Data ingestion completed successfully")
        logger.debug(f"Artifact: {data_ingestion_artifact}")
        return data_ingestion_artifact


        


        