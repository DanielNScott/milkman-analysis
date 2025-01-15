# Foraging Analyses
Analysis code for foraging experiment.

## Approach
- Subject data is read into a list ```subj[s] = data```
- Trial aggregations are inserted in a group table ```grp[subj_id, column]```
- Various further data are computed and appended as columns
- Policy-optimization and policy-model functions are run
- Everything of interest is plotted

## Files
- analy.py - Analysis code
- depr.py - Deprecated stuff, snippets
- plots.py - All plotting functions
- reads.py - The data-reading functions
- run.ipy - Script that runs everything

## Requires
- Matplotlib
- Numpy
- Scipy
- Statsmodels

