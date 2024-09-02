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

from mlProject.utils.common import save_object

@dataclass
class DataTranfomationConfig:
    preprocess_object_file_path=os.path.join('artifacts','model.pkl')
    
class DataTranfomation:
    def __init__(self) -> None:
        self.data_trainfomation_config=DataTranfomationConfig()
        
    def get_data_trnsfomation(self):
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
                    ('inputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('inputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder()),
                    ('scler',StandardScaler())
                ]
            )
            
            preprocess=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])
            return preprocess
        except Exception as e:
            raise CustomException(e,sys)    
            
    def initiate_get_data_transform(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            
            
            preprocess_object=self.get_data_trnsfomation()
            
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_data=train_data.drop(columns=[target_column_name],axis=1)
            input_feature_test_data=test_data.drop(columns=[target_column_name],axis=1)
            
            target_feature_train_data=train_data[target_column_name]
            target_feature_test_data=test_data[target_column_name]
            
            
            input_feature_train_arr=preprocess_object.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocess_object.transform(input_feature_test_data)
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_data)]
            
            logging.info('save pkl')
            
            save_object(
                file_path=self.data_trainfomation_config.preprocess_object_file_path,
                obj=preprocess_object
            )
            
            return (
                train_arr,test_arr,self.data_trainfomation_config.preprocess_object_file_path
            )
            
            
        except:
            pass
                


