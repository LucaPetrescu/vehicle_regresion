import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Utils:

    df = None

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)

    def show_head(self, n=5):
        return self.df.head(n)

    def count_values(self):
        return self.df.dtypes.value_counts()

    def get_length(self):
        return len(self.df)

    def remove_nan_columns(self, number_of_nan_elements):
        self.df = self.df.dropna(thresh=len(
            self.df) - number_of_nan_elements, axis=1)
        return self.df

    def get_nan_values(self):
        return self.df.isna().sum()

    def get_number_columns(self):
        return self.df.select_dtypes(include='number')

    def get_object_columns(self):
        return self.df.select_dtypes(include='object')

    def fill_nan_with_median(self):
        for column in self.df.columns:
            if self.df[column].dtype != 'object':
                median_value = self.df[column].median()
                self.df[column].fillna(median_value, inplace=True)

    @staticmethod
    def detect_outliers(df, k=1.5):
        outliers_by_column = {}
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            outliers = df[col][(df[col] < lower_bound) |
                               (df[col] > upper_bound)]
            if not outliers.empty:
                outliers_by_column[col] = outliers
        return outliers_by_column

    def show_correlation(self):
        plt.rcParams['font.family'] = 'monospace'
        plt.figure(figsize=(20, 20), dpi=500)

        # Filter out the "data" column
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_df = numeric_df.corr()
        corr_df = corr_df - np.diag(np.diag(corr_df))
        corr_df = corr_df[corr_df > 0.4]
        corr_df = corr_df.dropna(axis=0, how='all')
        corr_df = corr_df.dropna(axis=1, how='all')

        sns.heatmap(corr_df, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, linecolor='black',
                    square=False, cbar=True, cbar_kws={'orientation': 'vertical', 'shrink': 0.8, 'pad': 0.05})
        plt.title('Pearson Correlation Matrix', fontsize=20)
        plt.tight_layout()

    def fill_nan_with_frequent(self):
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                unique_values = self.df[column].apply(lambda x: tuple(
                    x) if isinstance(x, list) else x).unique().tolist()
                unique_values = set(str(val) for val in unique_values)
                most_frequent_value = self.df[column].mode()[0]
                print(most_frequent_value)
                # if unique_values == {'Da', 'Nu'}:
                #     print(f"For column '{column}', most frequent value is '{most_frequent_value}'")
                # else:
                #     print('Nu')
