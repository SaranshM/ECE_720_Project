import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# temp
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def calculate_iqr_outliers(self, data, feature):
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound
    
    def remove_outliers(self, df):
        outliers_indices = []

        for feature in df.columns:
            if df[feature].dtype in ['float64', 'int64']:
                lower_bound, upper_bound = self.calculate_iqr_outliers(df, feature)
                feature_outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].index
                outliers_indices.extend(feature_outliers)

        # Determine which rows have outliers in multiple columns
        from collections import Counter
        outlier_counts = Counter(outliers_indices)
        multiple_outliers = [k for k, v in outlier_counts.items() if v > 1]

        df = df.drop(multiple_outliers)
        return df
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            df = pd.read_csv('src/notebook/data/smoking.csv')
            logging.info("Read the dataset")

            df = df.drop(['ID'], axis = 1)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info("Renaming columns")

            new_column_names = {
                "age": "age",
                "height(cm)": "height",
                "weight(kg)": "weight",
                "waist(cm)": "waist",
                "eyesight(left)": "eyesight_left",
                "eyesight(right)": "eyesight_right",
                "hearing(left)": "hearing_left",
                "hearing(right)": "hearing_right",
                "systolic": "systolic",
                "relaxation": "relaxation",
                "fasting blood sugar": "fasting_blood_sugar",
                "Cholesterol": "cholesterol",
                "triglyceride": "triglyceride",
                "HDL": "hdl",
                "LDL": "ldl",
                "hemoglobin": "hemoglobin",
                "Urine protein": "urine_protein",
                "serum creatinine": "serum_creatinine",
                "AST": "ast",
                "ALT": "alt",
                "Gtp": "gtp",
                "dental caries": "dental_caries",
                "smoking": "smoker_status"
            }

            df.rename(columns = new_column_names, inplace = True)

            df = self.remove_outliers(df)

            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, random_state = 42, test_size = 0.15)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    train_data, test_data = data_ingestion_obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_file_path = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_file_path)


