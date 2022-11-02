import numpy as np 
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

def corr_plot(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

def quad_geo_histogram(df, column, log=False):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(2, 2, 1)
    sns.histplot(data=df, x=f'grunnkrets_id.{column}', log_scale=log)
    ax = fig.add_subplot(2, 2, 2)
    sns.histplot(data=df, x=f'delomrade.{column}', log_scale=log)
    ax = fig.add_subplot(2, 2, 3)
    sns.histplot(data=df, x=f'kommune.{column}', log_scale=log)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax = fig.add_subplot(2, 2, 4)
    sns.histplot(data=df, x=f'fylke.{column}', log_scale=log)
    plt.show()

def quad_geo_scatter(df, column, xscale='linear', yscale ='linear'):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(2, 2, 1)
    ax.set(xscale = xscale, yscale = yscale)
    sns.scatterplot(data=df, x=f"grunnkrets_id.{column}", y="revenue")
    ax = fig.add_subplot(2, 2, 2)
    ax.set(xscale = xscale, yscale = yscale)
    sns.scatterplot(data=df, x=f"delomrade.{column}", y="revenue")
    ax = fig.add_subplot(2, 2, 3)
    ax.set(xscale = xscale, yscale = yscale)
    sns.scatterplot(data=df, x=f"kommune.{column}", y="revenue")
    ax = fig.add_subplot(2, 2, 4)
    ax.set(xscale = xscale, yscale = yscale)
    sns.scatterplot(data=df, x=f"fylke.{column}", y="revenue")
    plt.show()