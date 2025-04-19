# Standard imports
import numpy  as np
import scipy  as sp
import scipy.stats as st
import pandas as pd

# Plotting
from matplotlib import pyplot as plt
import seaborn as sns

# Statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Plotting configs
save_path = './figs/'
dpi = 300
fmt = 'svg'

# Toggle for plotting
make_plots = False
plt.ion()

# Reproducibility
np.random.seed(0)