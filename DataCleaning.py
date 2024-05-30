import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataCleaning:

    # df = None

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)

    def show_head(self, df, n=5):
        return df.head(n)

    def count_values(df):
        return df.dtypes.value_counts()

    def get_length(self, df):
        return len(df)

    # def remove_nan_columns(self, number_of_nan_elements):
    #     self.df = self.df.dropna(thresh=len(
    #         self.df) - number_of_nan_elements, axis=1)
    #     return self.df

    def get_nan_values(self, df):
        return df.isna().sum()

    def get_number_columns(self, df):
        return df.select_dtypes(include='number')

    def get_object_columns(self, df):
        return df.select_dtypes(include='object')

    def fill_nan_with_median(self, df):
        for column in df.columns:
            if df[column].dtype != 'object':
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
        return df

    def show_correlation(self, df):
        plt.rcParams['font.family'] = 'monospace'
        plt.figure(figsize=(20, 20), dpi=500)
        numeric_df = df.select_dtypes(include=[np.number])
        corr_df = numeric_df.corr()
        corr_df = corr_df - np.diag(np.diag(corr_df))
        corr_df = corr_df[corr_df > 0.4]
        corr_df = corr_df.dropna(axis=0, how='all')
        corr_df = corr_df.dropna(axis=1, how='all')

        sns.heatmap(corr_df, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, linecolor='black',
                    square=False, cbar=True, cbar_kws={'orientation': 'vertical', 'shrink': 0.8, 'pad': 0.05})
        plt.title('Pearson Correlation Matrix', fontsize=20)
        plt.tight_layout()

    def delete_columns_with_nans(self, df, threshold=0.66):
        num_rows = len(df)
        for column in df.columns:
            nan_count = df[column].isna().sum()
            if nan_count >= threshold * num_rows:
                df.drop(columns=[column], inplace=True)
                print(
                    f"Deleted column '{column}' with {nan_count} NaN values (more than {threshold * 100}% of the rows)")
        return df

    def fill_nan_with_frequent(self, df):
        for column in df.select_dtypes(include=['object']).columns:
            if df[column].isna().any():
                most_frequent_value = df[column].mode()[0]
                df[column].fillna(most_frequent_value, inplace=True)
                print(
                    f"For column '{column}', filled NaN with '{most_frequent_value}'")
        return df

    def detect_categorical_data(self, df, max_unique_values=7):
        categorical_columns = []
        for column in df.select_dtypes(include=['object']).columns:
            unique_values_count = df[column].nunique()
            if unique_values_count <= max_unique_values:
                categorical_columns.append(column)
                print(
                    f"Column '{column}' is categorical with {unique_values_count} unique values.")
        return categorical_columns