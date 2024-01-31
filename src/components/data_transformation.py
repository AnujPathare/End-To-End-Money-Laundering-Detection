import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from category_encoders import CountEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Function to get all pickle files
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numeric_features = ['From Bank', 'To Bank', 'Amount Received', 'Amount Paid', 'Day', 'Hour', 'Minute']
            account_features = ['Sender Account', 'Receiver Account']
            payment_features = ['Receiving Currency', 'Payment Currency', 'Payment Format']

            logging.info("Numeric Features: {numeric_features}")
            logging.info("Categorical Features: {account_features + payment_features}")

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                    
            ])

            cat_pipeline_1 = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('count_encoder', CountEncoder())
            ])

            cat_pipeline_2 = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numeric_features),
                    ('cat_pipeline_1', cat_pipeline_1, account_features),
                    ('cat_pipeline_2', cat_pipeline_2, payment_features),
            ])
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Completed reading train and test data")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            # numeric_features = ['From Bank', 'To Bank', 'Amount Received', 'Amount Paid', 'Day', 'Hour', 'Minute']
            # account_features = ['Sender Account', 'Receiver Account']
            # payment_features = ['Receiving Currency', 'Payment Currency', 'Payment Format']

            target_column_name = 'Is Laundering'

            # Split train data into features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Split test data into features and target
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor object on training and testing dataframe")

            # Data Preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Resampling imbalanced data")

            # Resampling imbalanced data using SMOTE
            smote = SMOTE(random_state=42, k_neighbors=5)
            input_feature_train_arr_smote, target_feature_train_df_smote = smote.fit_resample(input_feature_train_arr, target_feature_train_df)
            
            # Final Training dataset
            target_feature_train_sparse = csr_matrix(target_feature_train_df_smote.values.reshape(-1, 1))
            train_arr = hstack([input_feature_train_arr_smote, target_feature_train_sparse])

            # Final Testing dataset
            target_feature_test_sparse = csr_matrix(target_feature_test_df.values.reshape(-1, 1))
            test_arr = hstack([input_feature_test_arr, target_feature_test_sparse])

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)