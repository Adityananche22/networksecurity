import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

import mlflow
import mlflow.sklearn
import dagshub

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)

from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score
)

# ============================================================
# DagsHub + MLflow Setup
# ============================================================

dagshub.init(
    repo_owner="Adityananche22",
    repo_name="networksecurity",
    mlflow=True
)

mlflow.set_experiment("networksecurity")


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        """
        Logs model and metrics to MLflow/DagsHub.
        """
        try:
            with mlflow.start_run():

                mlflow.log_metric(
                    "f1_score",
                    classification_metric.f1_score
                )

                mlflow.log_metric(
                    "precision_score",
                    classification_metric.precision_score
                )

                mlflow.log_metric(
                    "recall_score",
                    classification_metric.recall_score
                )

                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model"
                )

                logging.info("Successfully logged model to MLflow")

        except Exception as e:
            logging.warning(
                f"MLflow logging failed: {str(e)}"
            )

    def train_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):
        try:

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    solver="liblinear"
                ),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {

                "Decision Tree": {
                    "criterion": [
                        "gini",
                        "entropy",
                        "log_loss"
                    ]
                },

                "Random Forest": {
                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                "Gradient Boosting": {
                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],
                    "subsample": [
                        0.6,
                        0.7,
                        0.8,
                        0.9
                    ]
                },

                "Logistic Regression": {},

                "AdaBoost": {
                    "learning_rate": [
                        0.1,
                        0.01,
                        0.5,
                        0.001
                    ],
                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                }
            }

            logging.info("Evaluating models")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            logging.info(f"Model Report: {model_report}")

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[
                best_model_name
            ]

            best_model = models[
                best_model_name
            ]

            logging.info(
                f"Best Model Found: "
                f"{best_model_name} "
                f"with score: "
                f"{best_model_score}"
            )

            if best_model_score < 0.1:
                raise Exception(
                    "No acceptable model found."
                )

            # Train best model
            best_model.fit(
                X_train,
                y_train
            )

            # Training metrics
            y_train_pred = best_model.predict(
                X_train
            )

            train_metric = (
                get_classification_score(
                    y_true=y_train,
                    y_pred=y_train_pred
                )
            )

            # Testing metrics
            y_test_pred = best_model.predict(
                X_test
            )

            test_metric = (
                get_classification_score(
                    y_true=y_test,
                    y_pred=y_test_pred
                )
            )

            # MLflow logging
            self.track_mlflow(
                best_model=best_model,
                classification_metric=test_metric
            )

            logging.info(
                f"Train Metric: {train_metric}"
            )

            logging.info(
                f"Test Metric: {test_metric}"
            )

            # Load preprocessor
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            # Create final model
            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model
            )

            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            os.makedirs(
                model_dir_path,
                exist_ok=True
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=network_model
            )

            save_object("models_final/model.pkl", best_model)

            model_trainer_artifact = (
                ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    train_metric_artifact=train_metric,
                    test_metric_artifact=test_metric
                )
            )

            logging.info(
                f"Model Trainer Artifact: "
                f"{model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(
                e,
                sys
            )

    def initiate_model_trainer(
        self
    ) -> ModelTrainerArtifact:

        try:

            logging.info(
                "Loading transformed train and test arrays"
            )

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model_trainer_artifact = (
                self.train_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
            )

            logging.info(
                "Model training completed successfully"
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(
                e,
                sys
            )