import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Transformation:
    def __init__(self, df):
        self.df = df

    def add_cyclic_encoding(self, features_cycle):
        """
        Adds cyclic encoding (sin and cos transformations) for specified features.

        Args:
        features_cycle : list of dicts
            A list of dictionaries where each dictionary has 'feature' as the key 
            and 'cycle' (max value for the feature) as the value.

        Returns:
        DataFrame : The DataFrame with additional sin and cos columns for each feature.
        """
        for feature_cycle in features_cycle:
            feature = feature_cycle['feature']
            cycle = feature_cycle['cycle']
            self.df[f'{feature}_sin'] = np.sin(2 * np.pi * self.df[feature] / cycle)
            self.df[f'{feature}_cos'] = np.cos(2 * np.pi * self.df[feature] / cycle)
        return self.df

    def standardize_by_year(self, features):
        """
        Standardizes the specified features by year.
        
        Args:
        features : list
            A list of feature names to standardize.

        Returns:
        DataFrame : The DataFrame with standardized features.
        """
        for feature in features:
            self.df[f'{feature}_stdize'] = self.df.groupby('year')[feature].transform(lambda x: (x - x.mean()) / x.std())
        return self.df

    def log_transform_precipitation(self, feature):
        """
        Applies log transformation to the precipitation feature.
        
        Args:
        feature : str
            The name of the feature to apply log transformation to.
        
        Returns:
        DataFrame : The DataFrame with the transformed feature.
        """
        self.df[f'{feature}_log'] = np.log1p(self.df[feature])  # log(precipitation + 1)
        return self.df

    def normalize_data(self, features):
        """
        Normalizes the specified features to a range between 0 and 1.
        
        Args:
        features : list
            A list of feature names to normalize.

        Returns:
        DataFrame : The DataFrame with normalized features.
        """
        scaler = MinMaxScaler()
    
        for feature in features:
            self.df[f'{feature}_norm'] = scaler.fit_transform(self.df[[feature]])  # Apply normalization to each feature
        return self.df

    def add_lag_features(self, columns_to_lag, lag_period, timestamp):
        """
        Adds lag features for specified columns at given lag intervals.

        Args:
        columns_to_lag : list
            List of column names to apply lag features.
        lag_period : list
            List of lag intervals (in hours or days).
        timestamp : str
        The name of the column containing the timestamp. To ensure the lag values are calculated in chronological order.

        Returns:
        DataFrame : The DataFrame with lag features added.
        """
        self.df = self.df.sort_values(by=timestamp)

        for col in columns_to_lag:
            for lag in lag_period:
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)

        self.df.dropna(inplace=True)  # Drop rows with NaN values created by lagging
        return self.df

    def add_rolling_mean(self, features, window=6):
        """
        Applies a rolling mean to the specified features.

        Args:
        features : list
            List of feature names to apply the rolling mean.
        window : int
            The window size for the rolling mean (default is 6 hours).

        Returns:
        DataFrame : The DataFrame with rolling mean features added.
        """
        for feature in features:
            self.df[f'{feature}_rolling_{window}h'] = self.df[feature].rolling(window=window).mean()
            self.df[f'{feature}_rolling_{window}h'].fillna(method='bfill', inplace=True)

        return self.df

