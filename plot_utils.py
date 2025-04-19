from configs import *
from utils import robust_zscore

def plot_grouped_bars(array, title=None, xlabel=None, ylabel=None, xticklabels=None, legend=None, rotation=0, figsize=(6, 4), yerr=None, colors=None):
    """
    Plots a grouped bar plot for the given 2D NumPy array.
    
    Parameters:
    array (numpy.ndarray): 2D array where rows are groups and columns are categories.
    """
    # Number of groups and categories
    num_groups, num_categories = array.shape

    # Set up the bar width and positions
    bar_width = 0.8 / num_groups
    index = np.arange(num_categories)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars for each group
    for i in range(num_groups):
        xlocs = index + i * bar_width - bar_width/num_groups

        if yerr is not None:
            if colors == None:
                ax.bar(xlocs, array[i], bar_width, yerr=yerr[i])
            else:
                ax.bar(xlocs, array[i], bar_width, yerr=yerr[i], color=colors[i])

        else:
            if colors == None:
                ax.bar(xlocs, array[i], bar_width)
            else:
                ax.bar(xlocs, array[i], bar_width, color=colors[i])
                

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (num_groups - 1) / 2 - bar_width/num_groups)
    
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=rotation)

    if legend is not None:
        ax.legend(legend)

    plt.tight_layout()

    # Show plot
    plt.show()

def scatter_with_lm_fit(x, y, c, xlabel, ylabel, figsize=[3.3,3]):
    if type(x) == pd.DataFrame:
        x = x.reset_index(drop=True)

    if type(y) == pd.DataFrame:
        y = y.reset_index(drop=True)

    ns = len(y)
    no_x = np.where(x.isna())[0]
    no_y = np.where(y.isna())[0]
    drop = np.union1d(no_y, no_x)

    x = x.drop(drop).reset_index(drop=True)
    y = y.drop(drop).reset_index(drop=True)
    c = c.drop(drop).reset_index(drop=True)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    pred = model.predict(X)
    frame = model.get_prediction(X).summary_frame(alpha=0.05)

    # obs_ci field includes expected uncertainty in new data
    # mean_ci field is uncertainty in regression line itself

    lower_ci = frame['mean_ci_lower']
    upper_ci = frame['mean_ci_upper']

    if type(x) == pd.DataFrame:
        x = x.values.squeeze()

    inds = np.argsort(x)

    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=c, s=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(x[inds], pred[inds], 'k--')
    plt.fill_between(x[inds], lower_ci[inds], upper_ci[inds], alpha=0.2)
    plt.tight_layout()

    corr = sp.stats.pearsonr(x, y)

    print(model.summary())
    print('p-vals: ' + str(model.pvalues.values))
    print('Coeffs: ' + str(model.params.values))
    print('SE: '+ str(model.bse.values))
    print('Correlation: ' + str(corr.statistic))
    print('p-value: ' + str(corr.pvalue))
    print('t-values: ' + str(model.tvalues.values))


def plot_scree_cumsum(evals, figsize=[3.3,3], pc_names = None):
    plt.figure(figsize=figsize); 
    plt.plot(np.cumsum(evals)/sum(evals), '-o', zorder=10)
    if pc_names is not None:
        plt.xticks(np.arange(len(pc_names)), pc_names, rotation=30, ha='right')
    plt.grid()
    plt.tight_layout()


def plot_correlation_matrix(df, cnames, figsize=[3.3*3,3*3]):
    # Correlation matrix of original variables
    corr = np.corrcoef(df[cnames].to_numpy().T)
    plt.figure(figsize=figsize)
    ax = plt.subplot()
    sns.heatmap(corr, annot=True, fmt=".1f", cmap='viridis', ax=ax)
    plt.xticks(np.arange(len(cnames))+0.5, cnames, rotation=90, ha='left')
    plt.yticks(np.arange(len(cnames))+0.5, cnames, rotation=0 , ha='right')
    plt.tight_layout()


def plot_column_histograms(df):
    for col in df.columns:
        plt.figure(figsize=[6.6,3])
        plt.suptitle(col)

        plt.subplot(1,2,1)
        plt.hist(df[col], zorder=10)
        plt.title('Raw')
        plt.show()
        plt.grid()

        plt.subplot(1,2,2)
        try:
            x = robust_zscore(df[col], skew_correction=True)
        except:
            x = []
        plt.hist(x, zorder=10)
        plt.title('Robust Z-Score')
        plt.show()
        plt.grid()

        plt.tight_layout()
        plt.savefig('./data/figs/hist_'+col+'.png', dpi=dpi, format=fmt)
        plt.close()
