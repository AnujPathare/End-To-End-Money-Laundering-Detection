import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# Configuration settings: parameters such as source location, file formats, connection details, etc# 
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiated Data Ingestion")
        try:
            df = pd.read_csv('notebook/data/transactions.csv')

            df.rename(columns = {'Account':'Sender Account', 'Account.1':'Receiver Account'}, inplace = True)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)

            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
            df['Year'] = df['Timestamp'].dt.year
            df['Month'] = df['Timestamp'].dt.month
            df['Day'] = df['Timestamp'].dt.day
            df['Hour'] = df['Timestamp'].dt.hour
            df['Minute'] = df['Timestamp'].dt.minute
            df.drop(columns=['Year', 'Month', 'Timestamp'], inplace=True)

            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.33, stratify=df['Is Laundering'], random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )

        except Exception as e:
            raise CustomException(e, sys)
        