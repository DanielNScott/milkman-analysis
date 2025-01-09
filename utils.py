import numpy as np
from scipy import stats
import scipy  as sp
import statsmodels.api as sm

def compare_means(values_1, values_2, print_results=True, labels = ['Group 1', 'Group 2']):

    if len(values_2) == 1:
        # One-sample t-test: compare values_1 to a single point estimate (mean of values_2)
        t_stat, p_value = stats.ttest_1samp(values_1, values_2[0])
    else:
        # Perform independent t-test (two-sided by default)
        t_stat, p_value = stats.ttest_ind(values_1, values_2)

    # Calculate Cohen's d
    d_prime = cohen_d(values_1, values_2)

    # Output the results
    if print_results:
        print(f"Comparing {labels[0]} and {labels[1]}:")
        print(f"T-statistic: {t_stat:.2e}")
        print(f"P-value: {p_value:.2e}")
        print(f"Cohen's d: {d_prime:.2e}")
        print("")

    return p_value, d_prime


# Calculate Cohen's d for effect size
def cohen_d(x, y):
    # Calculate the means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate the pooled standard deviation
    if len(y) == 1:
        pooled_std  = np.std(x, ddof=1)
    else:
        pooled_std = np.sqrt(((len(x) - 1) * np.std(x, ddof=1) ** 2 + (len(y) - 1) * np.std(y, ddof=1) ** 2) / (len(x) + len(y) - 2))        

    # Compute Cohen's d
    return (mean_x - mean_y) / pooled_std


def get_rank(col):
    ns  = len(col)
    idx = np.argsort(col)
    
    subj = np.arange(0, ns)
    rank = np.arange(0, ns)
    
    subj[idx] = rank
    
    return subj



def enforce_float_cols(df):
    for col in df:
        df[col] = df[col].astype(float)
    return df


def jse(data, fn):
    nreps = len(data)
    fval  = fn(data)
    evals = np.zeros([nreps, *fval.shape])
    
    inds = np.arange(nreps)
    for i in range(0, nreps):
        sub = inds[inds !=i ]
        evals[i] = fn( data[sub] )
    
    se = np.sqrt( (nreps - 1)/nreps * np.sum( (evals - fval)**2 ,axis=0) )
    
    return se



def PCA(df, pos_const = True):
    vals = np.array(df)
    evals, evecs = np.linalg.eig(np.cov(vals.T))

    inds = np.flip(np.argsort(evals))
    evals = evals[inds]
    evecs = evecs[:,inds]

    if pos_const:
        evecs[:,0:1] = evecs[:,0:1] if evecs[0,0:1] > 0 else -evecs[:,0:1]

        if evecs[0,1] < 0:
            evecs[:,1] = -evecs[:,1]

        if evecs[2,2] < 0:
            evecs[:,2] = -evecs[:,2]

        if evecs[3,2] < 0:
            evecs[:,3] = -evecs[:,3]

    scores = (vals - np.mean(vals,axis=0)) @ evecs
    return evals, evecs, scores


def nanless(x):
    return x[~np.isnan(x)]

def get_complete_rows_only(df):
    return df.dropna(axis=0, how='any')

def subj_to_nan(subj, grp, droplist):
    grp.iloc[droplist] = np.nan
    for d in droplist: subj[d][:] = np.nan
    return subj, grp

# Wrapper for residualizing variables
def get_resid(y, X, disp = True):
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    if disp:
        print(res.summary())
    return res.resid


# Wrapper for the scipy linear model call
def run_lm(y, X, disp = True, zscore = False, robust = False, add_const = True):

    # Z-score data
    if zscore:
        X = sp.stats.zscore(X)
        y = sp.stats.zscore(y)
    
    if add_const:
        X = sm.add_constant(X)
    
    # Robust or standard OLS
    if robust:
        mod = sm.RLM(y, X, M = sm.robust.norms.HuberT())
    else:
        mod = sm.OLS(y, X)
    
    # Fit model
    res = mod.fit()
    
    # Tell user
    if disp:
        print(res.summary())
    
    return res

def robust_zscore(x, weight = 1):
    # Demand 1D input
    if len(x.shape) > 1:
        raise ValueError('Input must be a 1D array')
    
    # MAD can sometimes be very small, as w/ skewed data
    if weight > 1.0 or weight < 0.0:
        raise ValueError('Weight must be between 0 and 1')

    # Compute z-score
    z = x - np.median(nanless(x))
    mad = np.median(np.abs(nanless(z)))
    scale = weight*mad*1.48 + (1.0-weight)*np.std(nanless(z))

    # Warn if problems
    if scale == 0:
        print('Warning: Scale is zero in robust z-score.')
        return np.zeros_like(x)

    return z/scale

def robust_zscore_cols(df, weight = 0.5):
    for col in df.columns:
        df[col] = robust_zscore(df[col], weight)
    return df

def find_outliers_vec(x, thresh = 5):
    inds = np.where(np.abs(robust_zscore(x)) > thresh)[0].tolist()
    return np.sort(np.unique(inds)).tolist()

def find_outliers_df(df, thresh = 5):
    inds = []
    for col in df.columns:
        inds += find_outliers_vec(df[col], thresh)
    return np.sort(np.unique(inds)).tolist()

def iterative_outlier_pruning_vec(x, thresh = 5):
    if len(x.shape) > 1:
        raise ValueError('Input must be a 1D array')
    inds = find_outliers_vec(x, thresh)
    while len(inds) > 0:
        x = x.drop(inds, axis=0).reset_index(drop=True)
        inds = find_outliers_vec(x, thresh)
    return x

def iterative_outlier_pruning_df(df, thresh = 5):
    inds = find_outliers_df(df, thresh)
    while len(inds) > 0:
        df = df.drop(inds, axis=0).reset_index(drop=True)
        inds = find_outliers_df(df, thresh)
    return df


