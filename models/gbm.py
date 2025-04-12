import lightgbm as lgb
import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

class DataMerger:
    """
    Class to merge hourly and daily records into a single dataset.
    """
    def __init__(self, hourly_records, daily_records):
        """
        Initializes the DataMerger class with hourly and daily records.

        Args:
            hourly_records: List of dictionaries, JSON string, or DataFrame.
            daily_records: List of dictionaries, JSON string, or DataFrame.
        """
        # Handle different input types
        if isinstance(hourly_records, pd.DataFrame):
            self.hourly_data = hourly_records
        elif isinstance(hourly_records, (list, dict)):
            self.hourly_data = pd.DataFrame(hourly_records)
        elif isinstance(hourly_records, str):
            self.hourly_data = pd.read_json(hourly_records, orient='records')
        else:
            raise TypeError(f"Unsupported hourly_records type: {type(hourly_records)}")
        
        if isinstance(daily_records, pd.DataFrame):
            self.daily_data = daily_records
        elif isinstance(daily_records, (list, dict)):
            self.daily_data = pd.DataFrame(daily_records)
        elif isinstance(daily_records, str):
            self.daily_data = pd.read_json(daily_records, orient='records')
        else:
            raise TypeError(f"Unsupported daily_records type: {type(daily_records)}")
        
        # Normalize column names
        self.hourly_data.columns = [str(col).split(':')[-1].strip('"') for col in self.hourly_data.columns]
        self.daily_data.columns = [str(col).split(':')[-1].strip('"') for col in self.daily_data.columns]
        
        self.merged_data = pd.DataFrame()

    def merge_data(self):
        """
        Merges hourly data with daily data on the 'date' column and fills missing values.

        Returns:
            pd.DataFrame: Merged data frame containing both hourly and daily data.
        """
        # Ensure date columns are in datetime format
        self.hourly_data['date'] = pd.to_datetime(self.hourly_data['timestamp']).dt.date
        self.daily_data['date'] = pd.to_datetime(self.daily_data['date']).dt.date

        # Drop optional columns if present
        columns_to_drop = ['year', 'day_of_year', 'day_of_year_cos', 'day_of_year_sin']
        self.daily_data = self.daily_data.drop(columns=[col for col in columns_to_drop if col in self.daily_data.columns])
        
        # Merge the hourly data with daily data
        self.merged_data = pd.merge(self.hourly_data, self.daily_data, on='date', how='left')
        
        self.merged_data.fillna(method='ffill', inplace=True)
        self.merged_data.sort_values(by='timestamp', inplace=True)
        
        return self.merged_data
    
    
class LightGBMModel:
    """
    Class to build and train LightGBM models for regression tasks using time-series data.
    """
    def __init__(self,params=None):
        """
        Initializes the LightGBMModel class with the provided or default parameters.

        Args:
            params (dict, optional): Hyperparameters for LightGBM model. Defaults to None.
        """
        self.data = pd.DataFrame()
        self.target_columns = []
        self.feature_columns = []
        self.model = {}
        self.params=params if params is not None else{
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

    def get_data(self):
        """
        Returns the dataset.

        Returns:
            pd.DataFrame: The dataset.
        """
        return self.data

    def preprocess_data(self,target_columns, feature_columns):
        """
        Preprocesses the data by splitting it into training, validation, and test sets.

        Args:
            target_columns (list): List of target feature names.
            feature_columns (list): List of feature feature names.

        Returns:
            tuple: Split data into training, validation, and test sets.
        """
        self.target_columns = target_columns
        self.feature_columns = feature_columns

        
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values(by='date')

        # Split into train, validation and test sets
        train_end_date = '2022-12-31'  
        val_end_date = '2024-12-31'    

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
        """
        Trains the LightGBM model on the provided training data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target values.

        Returns:
            model: Trained LightGBM model.
        """
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100) 
        ]
        # Train the model
        model = lgb.train(self.params, 
            lgb_train, 
            valid_sets=[lgb_train,lgb_val], 
            num_boost_round=1000, 
            callbacks=callbacks
        )
        
        return model

    def predict(self, X_test, target):
        """
        Makes predictions using the trained model.

        Args:
            X_test (pd.DataFrame): Test features.
            target (str): The target column name for which predictions are to be made.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model:
            return self.model[target].predict(X_test, num_iteration=self.model[target].best_iteration)
        else:
            raise ValueError("Model has not been trained yet.")

    def evaluate(self, y_test, y_pred):
        """
        Evaluates the model performance using RMSE and R2 score.

        Args:
            y_test (pd.Series): Actual target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            tuple: RMSE and R2 scores.
        """
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        return rmse, r2
    
    def display_result(self, predictions, errors):
        """
        Displays the results for each target.

        Args:
            predictions (dict): Dictionary containing predictions for each target.
            errors (dict): Dictionary containing RMSE and R2 errors for each target.
        """
        for target in self.target_columns:
            print(f'Target: {target}\n prediction: {predictions[target]}\n error_rmse: {errors[target][0]}\n error_r2: {errors[target][1]}\n')
        
    def get_feature_importance(self, top_n=10):
        """
        Retrieves and displays the top N important features based on the LightGBM model.

        Args:
            top_n (int): Number of top features to display.

        Returns:
            list: List containing DataFrames with feature importance for each target model.
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

            # # Plot the feature importance
            # plt.figure(figsize=(10, 6))
            # feature_importance_df.head(top_n).plot(
            #     kind='barh', x='Feature', y='Importance', legend=False, figsize=(10, 6), color='teal'
            # )
            # plt.gca().invert_yaxis()
            # plt.title(f"Top {top_n} Feature Importances for Target: {target}")
            # plt.xlabel("Importance")
            # plt.ylabel("Feature")
            # plt.show()

            feature_importances.append((target, feature_importance_df))
        
        return feature_importances
    
    def time_series_cross_validation(self, features, target, n_splits=5):
        """
        Performs time-series cross-validation with the LightGBM model.

        Args:
            features (list): List of feature column names.
            target (str): The target column name.
            n_splits (int): Number of splits for cross-validation.

        Returns:
            dict: Dictionary containing RMSE for each fold for each model.
        """
        if not self.model:
            raise ValueError("Model dictionary is empty. Initialize models before running cross-validation.")
        
        X = self.data[features]  
        y = self.data[target] 
        overall_results = {}  # Store results for all models

        for target, model in self.model.items():
            print(f"\nCross-validating for model: {target}")
            cv_results = {}  # Store results for this model
            tscv = TimeSeriesSplit(n_splits=n_splits)

            for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                print(f"Processing Fold {fold + 1} for {target}_model...")

                # Split data into training and test sets
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Prepare LightGBM Datasets
                lgb_train = lgb.Dataset(X_train, label=y_train[target])
                lgb_test = lgb.Dataset(X_test, label=y_test[target], reference=lgb_train)

                # Train the model
                trained_model = lgb.train(
                    params=self.params,
                    train_set=lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    num_boost_round=1000,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(period=100)
                    ]
                )

                # Predict and calculate RMSE
                y_pred = trained_model.predict(X_test)
                rmse = mean_squared_error(y_test[target], y_pred) ** 0.5
                print(f"Fold {fold + 1} RMSE for {target}_model: {rmse:.4f}")
                cv_results[f"Fold_{fold + 1}"] = rmse

            # Aggregate results for this model
            overall_results[target] = {
                "Fold_RMSEs": cv_results,
                "Mean_RMSE": sum(cv_results.values()) / n_splits
            }
            print(f"\nAverage RMSE for {target}_model: {overall_results[target]['Mean_RMSE']:.4f}")

        return overall_results
    
    def save_models(self):
        """
        Saves all trained models to the 'models' folder in the project directory.
        """
        project_folder = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(project_folder, 'models')

        if not os.path.exists(save_path):
            os.makedirs(save_path)  
        for target, model in self.model.items():
            model_path = os.path.join(save_path, f'{target}_model.joblib')
            dump(model, model_path)

    def model_building(self,hourly_records, daily_records):
        """
        Builds and trains separate LightGBM models for each target variable.

        This method merges hourly and daily records into a single dataset, preprocesses 
        the data, and builds LightGBM models for each target variable. The models are then 
        trained, evaluated, and their results are displayed.

        Args:
            hourly_records (str): Path to the hourly records dataset (JSON file).
            daily_records (str): Path to the daily records dataset (JSON file).

        Returns:
            None
        """
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

        #feature_importance=self.get_feature_importance()
        #cross_val_overall_result=self.time_series_cross_validation(features_columns,target_columns)
        #print(cross_val_overall_result)

        self.save_models()
        return 