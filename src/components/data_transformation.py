import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class dataTransformation:
    def __init__(self):
        self.data_Transformation_Config = dataTransformationConfig()

    def get_data_transformer_object(self):
        try:
             numerical_columns = ["writing_score", "reading_score"]
             categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
             
             num_pipeline = Pipeline(
                 steps=[
                     ("Imputer",SimpleImputer(strategy="median")),
                     ("Scaler",StandardScaler())
                     
                 ]
             )

             cat_pipeline = Pipeline(
                 steps=[
                     ("Imputer",SimpleImputer(strategy="most_frequent")),
                     ("OneHotEncoder",OneHotEncoder()),
                     ("Scaler",StandardScaler(with_mean=False))
                 ]
             )

             preprocessor = ColumnTransformer(
                 
                 transformers=[
                     ("numericalPipeline",num_pipeline,numerical_columns),
                     ("categoricalPipeline",cat_pipeline,categorical_columns)
                 ]
             )

             return preprocessor
        except Exception as e:
            raise CustomException(e,sys) 

    def initaite_data_transformation(self,train_path,test_path):
        try:    

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Dataset has been saparated')
            logging.info("Obtaining preprocessing object")


            preprocessor_obj = self.get_data_transformer_object()
            target_col_name = "math_score"

            input_train_df = train_df.drop(columns=[target_col_name],axis=1)
            target_train_df = train_df[target_col_name]

            input_test_df = test_df.drop(columns=[target_col_name],axis=1)
            target_test_df = test_df[target_col_name]

            logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )
            
            input_feature_train=preprocessor_obj.fit_transform(input_train_df)
            input_feature_test=preprocessor_obj.transform(input_test_df)

            train_arr = np.c_[
                input_feature_train,np.array(target_train_df)
            ]

            test_arr = np.c_[
                input_feature_test,np.array(target_test_df)
            ]

            save_object(
                file_path = self.data_Transformation_Config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info(f"Saved preprocessing object.")

            return(
                train_arr,
                test_arr,
                self.data_Transformation_Config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

