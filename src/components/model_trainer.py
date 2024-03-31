import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Create train - test split")
            X_train, Y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, Y_test = test_arr[:, :-1], test_arr[:, -1]

            Y_train = Y_train.astype('int')
            Y_test = Y_test.astype('int')

            rf = RandomForestClassifier(bootstrap=True, max_depth=40, max_features="sqrt", min_samples_leaf=1, min_samples_split=2, n_estimators=800)

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = rf
            )

            
        except Exception as e:
            raise CustomException(e, sys)