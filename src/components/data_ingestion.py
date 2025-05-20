# Few library Imported

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import dataTransformation
from src.components.data_transformation import dataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    test_data_path: str = os.path.join('artifacts',"test.csv")
    train_data_path: str = os.path.join('artifacts',"train.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def initiatedataIngestion(self):
        try:
            df = pd.read_csv('StudentsPerformance.csv')
            logging.info('File has been read')
            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path),exist_ok=True)

            df.to_csv(self.ingestionConfig.raw_data_path,index=False,header=True)

            logging.info('Train Test Spilit Initated')

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestionConfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionConfig.test_data_path,index=False,header= True)

            logging.info('Train Test Split Completed')

            return(
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiatedataIngestion()

    data_transformation=dataTransformation()
    train_arr,test_arr,_ = data_transformation.initaite_data_transformation(train_data,test_data)
    model_trainer = ModelTrainer()
    print(model_trainer.initaite_model_trainer(train_arr,test_arr))
