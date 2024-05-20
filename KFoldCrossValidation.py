from typing import Any, Dict, List
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
# from sklearn.preprocessing import LabelEncoder

class KFoldCrossValidation:

    def __init__(self, df):
        self.df = df

    def leq_range_buckets(self, column: str, bucket_ranges: List[Any]) -> Dict[Any, pd.DataFrame]:
        buckets = {}

        for i, bucket_range in enumerate(bucket_ranges):
            if i == 0:
                buckets[f"<{bucket_range}"] = self.df[self.df[column] <= bucket_range]
            else:
                buckets[f"<{bucket_range}"] = self.df[(self.df[column] > bucket_ranges[i-1]) & (self.df[column] <= bucket_range)]
        
        total = sum([len(buckets[bucket]) for bucket in buckets])
        if total != len(self.df):
            buckets[f">{bucket_ranges[-1]}"] = self.df[self.df[column] > bucket_ranges[-1]]
        buckets["total"] = self.df
        return buckets
    
    def make_folds(self, buckets, n_splits: int = 5, shuffle: bool = False)-> Dict[str, KFold]:
        kfolds = {}
        for bucket_name, bucket_df in buckets.items():
            kf = KFold(n_splits=n_splits, shuffle=shuffle)
            kfolds[bucket_name] = (kf.split(bucket_df))
        return kfolds
    
    def get_train_test_folds(self, buckets: dict[str, pd.DataFrame], kfolds: dict[str, KFold], n_folds: int) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        train_folds: Dict[str, List[pd.DataFrame]] = {}
        test_folds: Dict[str, List[pd.DataFrame]] = {}
        
        bucket_names = set(buckets.keys())
        bucket_names.remove("total")
        for bucket_name in bucket_names:
            train_folds[bucket_name]: List[pd.DataFrame] = []
            test_folds[bucket_name]: List[pd.DataFrame] = []
            for idx, (train_idx, val_idx) in enumerate(kfolds[bucket_name]):
                train_indices = buckets[bucket_name].iloc[train_idx]
                test_indices = buckets[bucket_name].iloc[val_idx]
                train_folds[bucket_name].append(train_indices)
                test_folds[bucket_name].append(test_indices)

        train_folds_concat: List[pd.DataFrame] = []
        test_folds_concat: List[pd.DataFrame] = []
        for idx in range(n_folds):
            train_folds_concat.append(pd.concat([train_folds[bucket_name][idx] for bucket_name in bucket_names]))
            test_folds_concat.append(pd.concat([test_folds[bucket_name][idx] for bucket_name in bucket_names]))
        
        return train_folds_concat, test_folds_concat
    
    def plot_correlation_with_price(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()
        correlations = abs(correlations)
        target_correlations = correlations['pret']
        target_correlations = target_correlations.sort_values(ascending=False)
        removed_features = target_correlations[target_correlations < 0.15]
        removed_features = removed_features.index
        removed_features = list(removed_features)
        train_df = numeric_df.drop(removed_features, axis=1)
        correlation_matrix = train_df.corr()
        f, ax = plt.subplots(figsize=(36, 18))
        sns.heatmap(correlation_matrix, vmax=0.8, square=True)
        return train_df

    # def cramers_v(self, x, y):
    #     confusion_matrix = pd.crosstab(x, y)
    #     chi2 = ss.chi2_contingency(confusion_matrix)[0]
    #     n = confusion_matrix.sum().sum()
    #     phi2 = chi2/n
    #     r, k = confusion_matrix.shape
    #     phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    #     rcorr = r - ((r-1)**2)/(n-1)
    #     kcorr = k - ((k-1)**2)/(n-1)
    #     return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    # def calculate_cramers_v_matrix(self, df):
    #     cat_cols = df.select_dtypes(include=['object']).columns
    #     cramers_v_matrix = pd.DataFrame(np.zeros((len(cat_cols), len(cat_cols))), 
    #                                     index=cat_cols, columns=cat_cols)
    #     for col1 in cat_cols:
    #         for col2 in cat_cols:
    #             if col1 == col2:
    #                 cramers_v_matrix.loc[col1, col2] = 1
    #             else:
    #                 cramers_v_matrix.loc[col1, col2] = self.cramers_v(df[col1], df[col2])
    #     return cramers_v_matrix

    # def show_combined_correlation(self):
    #     plt.rcParams['font.family'] = 'monospace'
    #     plt.figure(figsize=(20, 20), dpi=500)

    #     numeric_df = self.df.select_dtypes(include=[np.number])
    #     categorical_df = self.df.select_dtypes(include=['object'])
    #     corr_numeric = numeric_df.corr()
        
    #     corr_categorical = self.calculate_cramers_v_matrix(self.df)
        
    #     combined_corr = pd.DataFrame(np.zeros((len(self.df.columns), len(self.df.columns))), 
    #                                  index=self.df.columns, columns=self.df.columns)
        
    #     for col in corr_numeric.columns:
    #         combined_corr.loc[col, corr_numeric.columns] = corr_numeric.loc[col, :]
        
    #     for col in corr_categorical.columns:
    #         combined_corr.loc[col, corr_categorical.columns] = corr_categorical.loc[col, :]
        
    #     sns.heatmap(combined_corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, linecolor='black',
    #                 square=False, cbar=True, cbar_kws={'orientation': 'vertical', 'shrink': 0.8, 'pad': 0.05})
    #     plt.title('Combined Correlation Matrix', fontsize=20)
    #     plt.tight_layout()
    #     plt.show()




        