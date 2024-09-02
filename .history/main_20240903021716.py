import pandas as pd
import os
import sys
from mlProject.logger import logging
from mlProject.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Enter the data ingestion part')
        try:
            # Use raw string for path to avoid escape sequence issues
            data = pd.read_csv(r'research\stud.csv')
            logging.info('Read the data')
            
            # Create directories if they do not exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            
            # Save the raw data
            data.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Save the main data')
            
            # Perform train-test split
            logging.info('Train-test split the main data')
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            
            # Save train and test data
            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info('Save the train and test data')
            
            logging.info('Complete the data ingestion part')
            
            # Return the paths
            return (self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
