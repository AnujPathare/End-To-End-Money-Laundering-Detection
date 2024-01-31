import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomException(e, sys)

# Class to map inputs given to HTML with backend
class CustomData:
    def __init__(self,
        From_Bank: int,
        Sender_Account: str,
        To_Bank: int,
        Receiver_Account: str,
        Amount_Received: float,
        Receiving_Currency: str,
        Amount_Paid: float,
        Payment_Currency: str,
        Payment_Format: str,
        Day: int,
        Hour: int,
        Minute: int
    ):
        self.From_Bank = From_Bank
        self.Sender_Account = Sender_Account
        self.To_Bank = To_Bank
        self.Receiver_Account = Receiver_Account
        self.Amount_Received = Amount_Received
        self.Receiving_Currency = Receiving_Currency
        self.Amount_Paid = Amount_Paid
        self.Payment_Currency = Payment_Currency
        self.Payment_Format = Payment_Format
        self.Day = Day
        self.Hour = Hour
        self.Minute = Minute

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "From Bank": [self.From_Bank],
                "Sender Account": [self.Sender_Account],
                "To Bank": [self.To_Bank],
                "Receiver Account": [self.Receiver_Account],
                "Amount Received": [self.Amount_Received],
                "Receiving Currency": [self.Receiving_Currency],
                "Amount Paid": [self.Amount_Paid],
                "Payment Currency": [self.Payment_Currency],
                "Payment Format": [self.Payment_Format],
                "Day": [self.Day],
                "Hour": [self.Hour],
                "Minute": [self.Minute]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)