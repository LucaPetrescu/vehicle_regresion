import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class EDA:

    df = None

    def __init__(self, df):
        self.df = df

    def show_distribution(self, df):
        f, ax = plt.subplots(figsize=(16, 8))
        sns.distplot(df['pret'], fit=norm)
        fig = plt.figure()

    def handle_non_positive(self, series):
        min_value = series.min()
        if min_value <= 0:
            series += (1 - min_value)
        return series

    def apply_logarithmic(self, df, columns):
        log_df = df[columns].copy()
        for column in columns:
            log_df[column] = self.handle_non_positive(log_df[column])
            log_df[column] = np.log(log_df[column])
        return log_df
    
    def convert_columns_to_numeric(self, df, columns):
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df
    
    def detect_outliers(self, df, k=1.5):
        
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        numeric_df = df.select_dtypes(include=['number'])
        outliers_by_column = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            outliers = numeric_df[col][(numeric_df[col] < lower_bound) |
                               (numeric_df[col] > upper_bound)]
            if not outliers.empty:
                outliers_by_column[col] = outliers
        return outliers_by_column
    
    def plot_outliers(self, df, outliers):
        for col, outliers in outliers.items():
            plt.figure(figsize=(10, 6))
            plt.scatter(df.index, df[col], label='Original Data')
            plt.scatter(outliers.index, outliers, color='r', label='Outliers')
            plt.title(f"Outliers in {col}")
            plt.legend()
            plt.show()

    def plot_outliers_scatter(self, df, outliers_by_column):
        for col, outliers in outliers_by_column.items():
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df.index, y=df[col], label='Data', color='blue')
            sns.scatterplot(x=outliers.index, y=outliers,
                            label='Outliers', color='red', marker='o')
            plt.title(f'Scatter Plot with Outliers for Column: {col}')
            plt.xlabel('Index')
            plt.ylabel(col)
            plt.legend()
            plt.show()

    def get_columns_names(self, df):
        return df.columns.tolist()
    
    def plot_graph(self, df, x_var, y_var):
        data = pd.concat([df[y_var], df[x_var]], axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200)
        ax.scatter(x=x_var, y=y_var, data=data, alpha=0.5, color="blue", s=10)
        ax.set_xlim(df[x_var].min(), df[x_var].max())
        ax.set_ylim(df[y_var].min(), df[y_var].max())
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(f"Correlation between {x_var} and {y_var}")
        ax.grid()
        ax.legend()
        plt.show()

