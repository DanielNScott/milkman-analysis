import numpy as np
from scipy import stats
import scipy  as sp
import statsmodels.api as sm

import copy
import re
import xml.etree.ElementTree as ET

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

## ----------------------------------------------------------------------- ##
##                       SVG Manipulation Functions                        ##
## ----------------------------------------------------------------------- ##
def parse_pixel_size(size_str):
    """
    Parse an SVG dimension string (e.g., '300px', '237.6pt', '3.5in')
    into approximate float (pixel) units. Returns None if parsing fails.
    """
    if not size_str:
        return None
    s = size_str.strip().lower()

    # Regex: numeric + optional unit
    match = re.match(r"^([\d\.]+)(px|pt|in|cm|mm)?$", s)
    if not match:
        return None

    numeric_val, unit = match.groups()
    val = float(numeric_val)

    # Convert to px
    if unit is None or unit == 'px':
        return val
    elif unit == 'pt':
        # 1 pt = 1/72 in, 1 in = 96 px => 1 pt = 1.3333 px
        return val * 96.0/72.0
    elif unit == 'in':
        return val * 96.0
    elif unit == 'cm':
        return val * 37.79527559
    elif unit == 'mm':
        return val * 3.779527559
    return None

def parse_viewbox(svg_element):
    """
    If the element has a 'viewBox' attribute like "0 0 400 300", return (width, height)
    from the last two numbers. Return None if not present or invalid.
    """
    viewbox = svg_element.attrib.get('viewBox')
    if not viewbox:
        return None
    # Typically "minX minY width height"
    parts = viewbox.strip().split()
    if len(parts) != 4:
        return None
    try:
        # We only care about the width, height portion
        vb_width = float(parts[2])
        vb_height = float(parts[3])
        return (vb_width, vb_height)
    except ValueError:
        return None

def combine_svgs(svg_paths, out_path, ncols=None, nrows=None):
    """
    Combine multiple SVGs into a single SVG, preserving vector data.
    - If a child has a viewBox, we interpret the child's size from that,
      ignoring 'width'/'height' in the child.
    - If no viewBox, we fall back on child's width/height attributes.
    - Then all cells are placed in a uniform grid, each cell sized to the
      first child's bounding box.

    # Example usage
    svg_files = ["./test.svg"]*6
    combine_svgs_same_size(svg_files, "combined.svg", ncols=3, nrows=2)
    """
    import math

    if not svg_paths:
        raise ValueError("No SVG paths provided.")

    # Parse first SVG to get the canonical child cell size
    first_tree = ET.parse(svg_paths[0])
    first_root = first_tree.getroot()

    # 1) Try child viewBox
    vb_dims = parse_viewbox(first_root)
    if vb_dims is not None:
        sub_width_px, sub_height_px = vb_dims
    else:
        # 2) Fallback: parse child's width/height
        w_str = first_root.attrib.get('width', '')
        h_str = first_root.attrib.get('height', '')
        sub_width_px = parse_pixel_size(w_str)
        sub_height_px = parse_pixel_size(h_str)
        if sub_width_px is None or sub_height_px is None:
            raise ValueError("Cannot determine sub-SVG size from first file")

    # Decide how many rows/cols to layout
    n_plots = len(svg_paths)
    if ncols is None and nrows is None:
        ncols = int(math.ceil(math.sqrt(n_plots)))
        nrows = int(math.ceil(n_plots / ncols))
    elif ncols is None:
        ncols = int(math.ceil(n_plots / nrows))
    elif nrows is None:
        nrows = int(math.ceil(n_plots / ncols))

    # Master <svg> total size in px (based on sub_width_px, sub_height_px)
    total_width = sub_width_px * ncols
    total_height = sub_height_px * nrows

    # Create a blank master <svg>
    master_str = f'''<svg width="{total_width}px" height="{total_height}px"
                      version="1.1"
                      xmlns="http://www.w3.org/2000/svg">
                     </svg>'''
    master_root = ET.fromstring(master_str)

    def transform_for_position(i):
        """Translate sub-SVG to row/col in the grid."""
        row = i // ncols
        col = i % ncols
        x = col * sub_width_px
        y = row * sub_height_px
        return f"translate({x},{y})"

    for i, path in enumerate(svg_paths):
        tree = ET.parse(path)
        child_svg = tree.getroot()
        sub_copy = copy.deepcopy(child_svg)

        # We'll remove the child's 'width', 'height', and 'viewBox' (if present),
        # so that it doesn't forcibly define its own size or scale.
        for attr in ['width', 'height', 'viewBox']:
            if attr in sub_copy.attrib:
                del sub_copy.attrib[attr]

        # Rename <svg> to <g> to avoid nested <svg> complexities
        sub_copy.tag = '{http://www.w3.org/2000/svg}g'

        # Place it
        sub_copy.set('transform', transform_for_position(i))

        master_root.append(sub_copy)

    ET.ElementTree(master_root).write(out_path, encoding='utf-8', xml_declaration=True)


