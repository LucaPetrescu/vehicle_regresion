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

    def delete_columns_with_nans(self, threshold=0.66):
        num_rows = len(self.df)
        for column in self.df.columns:
            nan_count = self.df[column].isna().sum()
            if nan_count >= threshold * num_rows:
                self.df.drop(columns=[column], inplace=True)
                print(
                    f"Deleted column '{column}' with {nan_count} NaN values (more than {threshold * 100}% of the rows)")

    def fill_nan_with_frequent(self):
        for column in self.df.select_dtypes(include=['object']).columns:
            if self.df[column].isna().any():
                most_frequent_value = self.df[column].mode()[0]
                self.df[column].fillna(most_frequent_value, inplace=True)
                print(
                    f"For column '{column}', filled NaN with '{most_frequent_value}'")

    def detect_categorical_data(self, max_unique_values=7):
        categorical_columns = []
        for column in self.df.select_dtypes(include=['object']).columns:
            # Get the number of unique values in the column
            unique_values_count = self.df[column].nunique()
            # Check if the number of unique values is less than or equal to max_unique_values
            if unique_values_count <= max_unique_values:
                categorical_columns.append(column)
                print(
                    f"Column '{column}' is categorical with {unique_values_count} unique values.")
        return categorical_columns

    @staticmethod
    def plot_outliers_scatter(df, outliers_by_column):
        for col, outliers in outliers_by_column.items():
            plt.figure(figsize=(10, 6))

            # Plot all points
            sns.scatterplot(x=df.index, y=df[col], label='Data', color='blue')

            # Highlight outliers
            sns.scatterplot(x=outliers.index, y=outliers,
                            label='Outliers', color='red', marker='o')

            # Set plot title and labels
            plt.title(f'Scatter Plot with Outliers for Column: {col}')
            plt.xlabel('Index')
            plt.ylabel(col)

            # Show legend
            plt.legend()

            # Show plot
            plt.show()
