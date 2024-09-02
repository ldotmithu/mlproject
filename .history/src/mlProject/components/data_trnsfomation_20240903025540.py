import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from mlProject.exception import CustomException
from mlProject.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTranfomationConfig:
    preprocess_object_file_path=os.path.join('artifacts','model.pkl')
    
class DataTranfomation:
    def __init__(self) -> None:
        self.data_trainfomation_config=DataTranfomationConfig()
        
    def det_data_trnsfomation(self):
        logging.info('enter DT')
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",]
            num_pipeline=Pipeline(
                steps=[
                    ('inputer',SimpleImputer(strategy='medi'))
                ]
            )
            
        except:
            pass    
            


