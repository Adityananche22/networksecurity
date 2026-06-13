import os
import sys
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constants.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS
)

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import (
    DataTransformationConfig
)

from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates preprocessing pipeline
        """
        try:
            logging.info(
                "Entered get_data_transformer_object method"
            )

            imputer = KNNImputer(
                **DATA_TRANSFORMATION_IMPUTER_PARAMS
            )

            logging.info(
                f"KNN Imputer initialized with params: "
                f"{DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )

            preprocessing_obj = Pipeline(
                steps=[
                    ("imputer", imputer)
                ]
            )

            logging.info(
                "Pipeline created successfully"
            )

            return preprocessing_obj

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(
        self
    ) -> DataTransformationArtifact:

        try:
            logging.info(
                "Entered initiate_data_transformation method"
            )

            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )

            test_df = self.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            logging.info(
                f"Train DataFrame Shape: {train_df.shape}"
            )

            logging.info(
                f"Test DataFrame Shape: {test_df.shape}"
            )

            # =========================
            # Train Data
            # =========================

            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN],
                axis=1
            )

            target_feature_train_df = train_df[
                TARGET_COLUMN
            ].replace(-1, 0)

            # =========================
            # Test Data
            # =========================

            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN],
                axis=1
            )

            target_feature_test_df = test_df[
                TARGET_COLUMN
            ].replace(-1, 0)

            logging.info(
                "Obtaining preprocessing object"
            )

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            logging.info(
                "Applying preprocessing on train data"
            )

            transformed_input_train_feature = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            logging.info(
                "Applying preprocessing on test data"
            )

            transformed_input_test_feature = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            train_arr = np.c_[
                transformed_input_train_feature,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                transformed_input_test_feature,
                np.array(target_feature_test_df)
            ]

            logging.info(
                "Saving transformed train array"
            )

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            logging.info(
                "Saving transformed test array"
            )

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            logging.info(
                "Saving preprocessing object"
            )

            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessing_obj
            )

            # Save preprocessor separately
            os.makedirs(
                "models_final",
                exist_ok=True
            )

            save_object(
                file_path="models_final/preprocessor.pkl",
                obj=preprocessing_obj
            )

            data_transformation_artifact = (
                DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
            )

            logging.info(
                f"Data Transformation Artifact: "
                f"{data_transformation_artifact}"
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)