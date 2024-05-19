import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt


class EDA:

    df = None

    def __init__(self, df):
        self.df = df

    def show_distribution(self):
        f, ax = plt.subplots(figsize=(16, 8))
        sns.distplot(self.df['pret'], fit=norm)

        fig = plt.figure()
