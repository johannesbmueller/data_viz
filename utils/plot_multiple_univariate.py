import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

def plot_multiple_univariate_plots(df, column):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], bins=15, kde=False)
    plt.title(f'Histogram of {column}')
    plt.xlabel(f'{column}')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.kdeplot(df[column], fill=True)
    plt.title(f'KDE Plot of {column}')
    plt.xlabel(f'{column}')
    plt.ylabel('Density')
    plt.show()

    plt.figure(figsize=(2, 6))
    sns.boxplot(y=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(f'{column}')
    plt.show()

    plt.figure(figsize=(2, 6))
    sns.violinplot(y=df[column])
    plt.title(f'Violin Plot of {column}')
    plt.xlabel(f'{column}')
    plt.show()

    plt.figure(figsize=(4, 6))
    sns.swarmplot(x=np.zeros(len(df)), y=df[column], size=4)
    plt.title(f'Swarm Plot of {column}')
    plt.ylabel(f'{column}')
    plt.xticks([])  # Remove x-axis labels
    plt.show()

    plt.figure(figsize=(6, 4))
    stats.probplot(df[column], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {column}')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Ordered Values')
    plt.show()
