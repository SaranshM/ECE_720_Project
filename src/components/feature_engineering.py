# import sys
# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import PolynomialFeatures
# from src.exception import CustomException
# from src.logger import logging
# from dataclasses import dataclass

# @dataclass
# class FeatureEngineeringConfig:


# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.scaler = MinMaxScaler()
        # Float columns identified for scaling; this might need updates based on your specific dataset
        self.float_cols = None  

    def fit(self, X, y=None):
        # Identifying float columns for scaling
        self.float_cols = X.select_dtypes(include=['float64']).columns
        
        # Fitting polynomial features transformer
        self.poly.fit(X[['height', 'weight', 'waist']])
        
        # Fitting the scaler on identified float columns
        self.scaler.fit(X[self.float_cols])
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # Step 1: Direct Feature Creation
        X['bmi'] = X['weight'] / ((X['height'] / 100) ** 2)
        X['waist_to_height_ratio'] = X['waist'] / X['height']
        X['bp_index'] = X['systolic'] / X['relaxation']
        X['age_weight_interaction'] = X['age'] * X['weight']
        
        # Step 2: Polynomial Features
        poly_features = self.poly.transform(X[['height', 'weight', 'waist']])
        poly_feature_names = self.poly.get_feature_names_out(['height', 'weight', 'waist'])
        for i, name in enumerate(poly_feature_names):
            X[name] = poly_features[:, i]
        
        # Step 3: Binning Age
        X['age_group'] = pd.cut(X['age'], bins=[0, 18, 35, 65, 100], labels=['child', 'young_adult', 'adult', 'senior'])
        
        # Step 4: Log Transformations
        skewed_features = ['fasting_blood_sugar', 'cholesterol', 'triglyceride', 'hdl', 'ldl', 'ast', 'alt', 'gtp']
        for feature in skewed_features:
            X[f'log_{feature}'] = np.log1p(X[feature])
        
        # Step 5: Encoding Categorical Variables (including binned 'age_group')
        # Note: Ensure 'gender', 'oral', 'tartar', 'age_group' are the categorical columns in your dataset needing encoding
        X = pd.get_dummies(X, columns=['gender', 'oral', 'tartar', 'age_group'])
        
        # Step 6 & 7: Creating Indexes and Interactions
        X['health_risk_index'] = (X['log_triglyceride'] + X['log_cholesterol'] + X['log_ast'] + X['log_alt'] + X['log_gtp']) / 5
        X['metabolic_index'] = (X['waist'] + X['log_fasting_blood_sugar'] + (X['systolic'] + X['relaxation']) / 2) / 3
        X['waist_gender_interaction'] = X['waist'] * X['gender_M']  # Assuming 'gender_M' is created by get_dummies
        X['age_health_risk_interaction'] = X['age'] * X['health_risk_index']
        
        # Step 8: Binning 'health_risk_index' and Encoding
        X['health_risk_category'] = pd.cut(X['health_risk_index'], bins=3, labels=['low', 'medium', 'high'])
        X = pd.get_dummies(X, columns=['health_risk_category'])
        
        # Step 9: Scaling
        if self.float_cols is not None:  # Check to prevent scaling of non-existent columns
            X[self.float_cols] = self.scaler.transform(X[self.float_cols])
        
        return X
