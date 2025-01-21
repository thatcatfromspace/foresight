import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class DataMerger:
    def __init__(self, hourly_records, daily_records):
        self.hourly_data = pd.read_json(hourly_records, orient='records')
        self.daily_data = pd.read_json(daily_records, orient='records')

    def merge_data(self):
        # Ensure date columns are in datetime format
        self.hourly_data['date'] = pd.to_datetime(self.hourly_data['timestamp']).dt.date
        self.daily_data['date'] = pd.to_datetime(self.daily_data['date']).dt.date

        # Merge the hourly data with daily data
        self.daily_data = self.daily_data.drop(columns=['year', 'day_of_year', 'day_of_year_cos', 'day_of_year_sin'])
        merged_data = pd.merge(self.hourly_data, self.daily_data, on='date', how='left')
        
        # Handle missing values (forward fill)
        merged_data.fillna(method='ffill', inplace=True)
        #merged_data.to_csv('merged_data.csv', index=False)

        return merged_data
    
class LightGBMModel:
    def __init__(self):
        """
        data: pandas DataFrame containing the merged data
        target_column: name of the target column
        feature_columns: list of feature columns
        """
        
        self.data = pd.DataFrame()
        # self.target_column = target_column
        # self.feature_columns = feature_columns
        self.target_columns = []
        self.feature_columns = []
        self.model = {}

    def preprocess_data(self,target_columns, feature_columns):
        self.target_columns = target_columns
        self.feature_columns = feature_columns

        # Split into train, validation and test sets
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values(by='date')

        train_end_date = '2022-12-31'  
        val_end_date = '2023-12-31'    


        train_data = self.data[self.data['date'] <= train_end_date]
        val_data = self.data[(self.data['date'] > train_end_date) & (self.data['date'] <= val_end_date)]
        test_data = self.data[self.data['date'] > val_end_date]

        X_train = train_data[self.feature_columns] 
        y_train = train_data[self.target_columns]

        X_val = val_data[self.feature_columns]
        y_val = val_data[self.target_columns]

        X_test = test_data[self.feature_columns]
        y_test = test_data[self.target_columns]
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, X_train, y_train, X_val, y_val):
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        # Hyperparameters
        params = {
            'objective': 'regression',         # For regression problem
            'metric': 'rmse',                  # Root Mean Squared Error 
            'boosting_type': 'gbdt',           # Gradient Boosting Decision Tree
            'num_leaves': 31,                  # maximum number of leaves in one tree
            'learning_rate': 0.05, 
            'n_estimators': 1000,              # Number of boosting iterations
            'max_depth': 7,                    # Maximum depth to avoid overfitting
            'subsample': 0.9,                  # Use 90% of the data for each tree 
            'colsample_bytree': 0.8,           # Use 80% of features per tree
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100) 
        ]
        # Train the model
        model = lgb.train(params, 
            lgb_train, 
            valid_sets=[lgb_train,lgb_val], 
            num_boost_round=1000, 
            callbacks=callbacks
        )
        
        return model

    def predict(self, X_test, target):
        if self.model:
            return self.model[target].predict(X_test, num_iteration=self.model[target].best_iteration)
        else:
            raise ValueError("Model has not been trained yet.")

    def evaluate(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        return rmse, r2
    
    def display_result(self, predictions, errors):
        for target in self.target_columns:
            print(f'Target: {target}\n prediction: {predictions[target]}\n error_rmse: {errors[target][0]}\n error_r2: {errors[target][1]}\n')
        
    def get_feature_importance(self, top_n=10):
        """
        Retrieve and visualize the top N important features based on LightGBM model.

        Args:
            top_n (int): Number of top features to display.

        Returns:
            feature_importance_df (pd.DataFrame): A DataFrame containing feature names and their importance scores.
        """
        if not self.model:
            raise ValueError("Model has not been trained yet. Train the model first to get feature importances.")

        feature_importances = []
        
        for target, model in self.model.items():
            # Retrieve feature importances from the model
            importance = model.feature_importance(importance_type='split')
            feature_names = model.feature_name()
            
            # Create a DataFrame for feature importance
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)

            # Visualize the top N important features
            print(f"\nTop {top_n} Features for Target: {target}")
            print(feature_importance_df.head(top_n))

            # Plot the feature importance
            plt.figure(figsize=(10, 6))
            feature_importance_df.head(top_n).plot(
                kind='barh', x='Feature', y='Importance', legend=False, figsize=(10, 6), color='teal'
            )
            plt.gca().invert_yaxis()
            plt.title(f"Top {top_n} Feature Importances for Target: {target}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.show()

            feature_importances.append((target, feature_importance_df))
        
        return feature_importances
    
    def model_building(self,hourly_records, daily_records):
        merger=DataMerger(hourly_records,daily_records)
        self.data = merger.merge_data()

        target_columns=['temperature_2m','precipitation','sunrise_numeric_sin','sunrise_numeric_cos','sunset_numeric_sin','sunset_numeric_cos']
        date_time_col=['timestamp','date','sunrise','sunset']
        features_columns = [col for col in self.data.columns if col not in target_columns+date_time_col]
        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess_data(target_columns,features_columns)
        print("PREPROCESSING DONE\n")
        predictions_dict = {}  # Dictionary to store predictions for each target
        error_dict = {}
        for target in target_columns: # build a seperate model for each target
            model=self.train(X_train, y_train[target], X_val, y_val[target])
            self.model[target]=model
            print(f"TRAINING TARGET {target} DONE\n")

            y_pred= self.predict(X_test,target)
            print(f"PRDICTING TARGET {target} DONE\n")

            error_rmse, error_r2= self.evaluate(y_test[target], y_pred)
            print(f"EVALUATING TARGET {target} DONE\n")

            predictions_dict[target]=y_pred
            error_dict[target]=[error_rmse, error_r2]
        self.display_result(predictions_dict,error_dict)
        feature_importance=self.get_feature_importance()
        return
    