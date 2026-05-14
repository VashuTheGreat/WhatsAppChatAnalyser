import os
PIPELINE_NAME="whatsApp_chat_analyser"
ARTIFACT_DIR="artifacts"



# =================== Data Ingestion===========================
DATA_INGESTION_DIR_NAME="ingestion"
DATA_INGESTION_FEATURE_STORE_DIR="features"
FILE_NAME="data.csv"


DATA_INGESTION_INGESTED_DIR="ingested_data"
TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"
# Split ration
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO=0.2

# =================== Data Validation ===========================
DATA_VALIDATION_DIR_NAME="validation"
DATA_VALIDATION_REPORT_FILE_NAME="report.yaml"
DATA_VALIDATION_SCHEMA_FILE_PATH=os.path.join("config", "data_validation.yml")

# =================== Data Transformation ===========================
DATA_TRANSFORMATION_DIR_NAME="transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR="transformed"
DATA_TRANSFORMATION_TRAIN_FILE_NAME="train.csv"
DATA_TRANSFORMATION_TEST_FILE_NAME="test.csv"

# =================== Data Analyser ===========================
DATA_ANALYSER_DIR_NAME="analysis"
