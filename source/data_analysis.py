import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def correlation_heatmap(df):
    f, ax = plt.subplots(figsize=[14, 8])
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.show()


def pair_plot(df):
    sns.pairplot(df)
    plt.show()