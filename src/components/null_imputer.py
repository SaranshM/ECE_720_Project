import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class NullImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Imputers
        self.numerical_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        
        self.numerical_columns = ['age', 'height', 'weight', 'waist', 'eyesight_left',
       'eyesight_right', 'hearing_left', 'hearing_right', 'systolic',
       'relaxation', 'fasting_blood_sugar', 'cholesterol', 'triglyceride',
       'hdl', 'ldl', 'hemoglobin', 'urine_protein', 'serum_creatinine', 'ast',
       'alt', 'gtp', 'dental_caries']
        self.categorical_columns = ['gender', 'oral', 'tartar']

    def fit(self, X, y=None):
        if len(self.numerical_columns) > 0:
            self.numerical_imputer.fit(X[self.numerical_columns])
            
        if len(self.categorical_columns) > 0:
            self.categorical_imputer.fit(X[self.categorical_columns])
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # Apply the imputations
        if len(self.numerical_columns) > 0:
            X[self.numerical_columns] = self.numerical_imputer.transform(X[self.numerical_columns])
            
        if len(self.categorical_columns) > 0:
            X[self.categorical_columns] = self.categorical_imputer.transform(X[self.categorical_columns])
        
        return X
