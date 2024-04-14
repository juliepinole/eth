import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import random


def images_subplots(
        img_data: list,
        ncols: int = 2,
        nrows: int = 2,
        figsize: tuple = (10, 10),
        list_all_labels: list = None,
        labels: list = ['Pneumonia', 'Normal'],
        labels_mapping: dict = {
            'Pneumonia': 0,
            'Normal': 1,
        }

):
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )
    count_labels = pd.Series(list_all_labels).value_counts()
    index_pneumonia = [i for i, n in enumerate(list_all_labels) if n == 'Pneumonia']
    index_normal = [i for i, n in enumerate(list_all_labels) if n == 'Normal']
    index_outcomes = {
        'Pneumonia': index_pneumonia,
        'Normal': index_normal,
    }
    for col_idx, outcome in enumerate(labels):
        n_imgs = count_labels[outcome]
        items_to_plot = random.sample(range(0,n_imgs), nrows)
        print(f'items_to_plot are {items_to_plot}')
        for row_idx, item in enumerate(items_to_plot):
            i, j = row_idx, col_idx
            axs[i,j].imshow(img_data[index_outcomes[outcome][item]][0], cmap='gray')
            axs[i,j].axis('off')
            axs[i,j].set_title(outcome)

    fig.tight_layout()
    fig.show()


# Inspired from https://github.com/yohann84L/plot_metric/blob/master/plot_metric/functions.py
def custom_roc_curve(
        fpr,
        tpr,
        thresholds_values,
        thresholds_to_plot: list = [0.5],
        figsize: tuple = (7, 7),
        fontsize: dict = {
            'ax_title': 12,
        },
        dataset_type: str = 'train',
        auc_performance: float = None,
        c_roc_curve='black',
        c_random_guess='red',
        c_thresh_lines='black',
        ls_roc_curve='-',
        ls_thresh_lines=':',
        ls_random_guess='--',
        loc_legend='lower right',
        plot_threshold=True, linewidth=2, y_text_margin=0.05, x_text_margin=0.2,
):

    fig, ax = plt.subplots(figsize=figsize, squeeze=True)
    ax.set_title(
        f'ROC curve, AUC = {auc_performance:.2f}, dataset = {dataset_type}', fontsize=fontsize['ax_title']
        )

    # Plot roc curve
    ax.plot(fpr, tpr, color=c_roc_curve,
                lw=linewidth, label=f'ROC curve (area = {auc_performance:.2f})', linestyle=ls_roc_curve)

    # Plot reference line
    ax.plot([0, 1], [0, 1], color=c_random_guess, lw=linewidth, linestyle=ls_random_guess, label="Random guess")

    # Plot Thresholds
    if plot_threshold:
        for t in thresholds_to_plot:
            # Compute the y & x axis to trace the threshold
            idx_thresh, idy_thresh = (
                fpr[np.argmin(abs(thresholds_values - t))],
                tpr[np.argmin(abs(thresholds_values - t))]
            )
            # Plot vertical and horizontal line
            plt.axhline(y=idy_thresh, color=c_thresh_lines, linestyle=ls_thresh_lines, lw=linewidth)
            plt.axvline(x=idx_thresh, color=c_thresh_lines, linestyle=ls_thresh_lines, lw=linewidth)
            # Plot text threshold
            if idx_thresh > 0.5 and idy_thresh > 0.5:
                plt.text(x=idx_thresh - x_text_margin, y=idy_thresh - y_text_margin,
                            s='Threshold : {:.1f}'.format(t))
            elif idx_thresh <= 0.5 and idy_thresh <= 0.5:
                plt.text(x=idx_thresh + x_text_margin, y=idy_thresh + y_text_margin,
                            s='Threshold : {:.1f}'.format(t))
            elif idx_thresh <= 0.5 < idy_thresh:
                plt.text(x=idx_thresh + x_text_margin, y=idy_thresh - y_text_margin,
                            s='Threshold : {:.1f}'.format(t))
            elif idx_thresh > 0.5 >= idy_thresh:
                plt.text(x=idx_thresh - x_text_margin, y=idy_thresh + y_text_margin,
                            s='Threshold : {:.1f}'.format(t))

            # Plot redpoint of threshold on the ROC curve
            plt.plot(idx_thresh, idy_thresh, 'ro')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # manipulate
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals], fontsize=9)
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals], fontsize=9)
    ax.legend(loc=loc_legend)
    


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
