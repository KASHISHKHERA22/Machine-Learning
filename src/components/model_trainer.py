import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.components.data_transformation import dataTransformation
from src.components.data_transformation import dataTransformationConfig

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initaite_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Initaited")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Random Forest":RandomForestRegressor(),
                "XGboost":XGBRegressor(),
                "Catboost":CatBoostRegressor()
            }

            model_report: dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            logging.info("Best Model Score Initaited")
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info("best model found")
            print(best_model)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)

            r2Score = r2_score(y_test,predicted)
            print(r2Score)
            
        except Exception as e:
            raise CustomException(e,sys)