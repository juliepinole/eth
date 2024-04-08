import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

def dist_mult_plots(
    df: pd.DataFrame(),
    cols: list = ['Age'],
    figsize: tuple = (7, 7),
    ncols: int = 2,
    bar_plot: bool = False,
    fontsize: dict = {
        'ax_title': 12,
    },
    custom_bins: dict = None,
    **kwargs,
    ):
    n_var = len(cols)
    nrows, r = divmod(n_var, ncols)
    if r > 0:
        nrows += 1
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )
    for item, col in enumerate(cols):
        i, j = divmod(item, ncols)
        if bar_plot:
            to_plot = df[col].astype(str) if is_numeric_dtype(df[col]) else df[col]
            series_frequency = to_plot.value_counts(dropna=False)
            axs[i,j].bar(
                series_frequency.index,
                series_frequency.values,
                label=series_frequency.index,
            )
        else:
            bins = 'auto'
            if custom_bins is not None and col in custom_bins:
                bins = custom_bins[col]
            sns.histplot(
                data=df,
                x=col,
                ax=axs[i,j],
                bins=bins,
                kde=True,
                common_bins=False,
                common_norm=False,
            )
        
        axs[i,j].set_title(
            col,
            fontsize=fontsize['ax_title'],
        )
        axs[i,j].set_xlabel('')
        axs[i,j].set_ylabel('')

    while j < ncols - 1:
        j += 1
        axs[i,j].set_visible(False)
    fig.tight_layout()
    fig.show()

def hist_multiple_var_single_plot(
    df: pd.DataFrame(),
    cols: list = ['Age'],
    figsize: tuple = (7, 7),
    **kwargs,
    ):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=figsize,
        squeeze=True,
    )
    for col in cols:
        ax.hist(
            df[col],
            label=col,
            # **kwargs,
        )
    ax.legend(cols)
    ax.set_title( ",".join(cols))
    # print(**kwargs)
    fig.show()
