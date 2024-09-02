from dataclasses import dataclass
import os,sys
from mlProject.logger import logging
from mlProject.exception import CustomException

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
                "test_preparation_course",
            ]
        except:
            pass    
            


