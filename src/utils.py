import os
import sys

import dill
import pickle

from sklearn.metrics import recall_score

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info(f'Exception occured during saving {obj} object')
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            
            # Train model
            model.fit(X_train, y_train)

            # Predict on Test data
            y_test_pred = model.predict(X_test)

            # accuracy = accuracy_score(y_test, y_test_pred)
            # precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            # f1_src = f1_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] =  recall

        return report
    
    except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)