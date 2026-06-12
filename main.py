from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    TrainingPipelineConfig,
    DataTransformationConfig
)

import sys


if __name__ == "__main__":
    try:
        # Training Pipeline Config
        training_pipeline_config = TrainingPipelineConfig()

        # ==========================
        # DATA INGESTION
        # ==========================
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )

        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config
        )

        logging.info("Initiating Data Ingestion")

        data_ingestion_artifact = (
            data_ingestion.initiate_data_ingestion()
        )

        print("Data Ingestion Artifact:")
        print(data_ingestion_artifact)

        # ==========================
        # DATA VALIDATION
        # ==========================
        data_validation_config = DataValidationConfig(
            training_pipeline_config=training_pipeline_config
        )

        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )

        logging.info("Initiating Data Validation")

        data_validation_artifact = (
            data_validation.initiate_data_validation()
        )

        logging.info("Data Validation Completed")

        print("Data Validation Artifact:")
        print(data_validation_artifact)

        # ==========================
        # DATA TRANSFORMATION
        # ==========================
        logging.info("Data Transformation Started")

        data_transformation_config = DataTransformationConfig(
            training_pipeline_config=training_pipeline_config
        )

        # IMPORTANT:
        # Pass data_validation_artifact here
        Data_Transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )

        data_transformation_artifact = (
            Data_Transformation.initiate_data_transformation()
        )

        print("Data Transformation Artifact:")
        print(data_transformation_artifact)

        logging.info("Data Transformation Completed")

    except Exception as e:
        raise NetworkSecurityException(e, sys)