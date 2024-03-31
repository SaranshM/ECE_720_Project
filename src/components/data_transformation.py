import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from src.exception import CustomException
from src.logger import logging
import os
from sklearn.pipeline import Pipeline
from src.components.null_imputer import NullImputer
from src.components.feature_engineering import FeatureEngineering
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self, columns):
        try:
            df_pipeline = Pipeline(
                steps = [
                    ("imputer", NullImputer()),
                    ("feat_engineering", FeatureEngineering())
                ]
            )

            preprocesor = ColumnTransformer(
                [
                    ("df_pipeline", df_pipeline, columns)
                ]
            )

            return preprocesor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
        
            logging.info("Read train and test data completed")

            target_column_name = "smoker_status"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(list(input_feature_train_df.columns))
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)