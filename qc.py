from matplotlib import pyplot as plt
from itertools import product
from utils import *
from reads import drop_subjects

def quality_controls(subj, grp, plot=True, thresh=3, drop=True):
    # Check for reasonable RTs in RT task
    grp['rtt_qc'] = np.abs(robust_zscore(grp['rtt_rt_med'])) < thresh

    # Subjects with missing data (usually age or from ToL perfect score)
    grp['inc_qc'] = grp.isna().sum(axis=1) == 0

    # Any non-foraging-task QC failures
    cols = [c + '_qc' for c in ['rtt', 'aes', 'phq']]
    grp['exg_qc'] = grp[cols].all(axis=1)

    # Foraging task quality controls
    grp['rew_qc'] = np.abs(robust_zscore(grp['rew_tot'])) < thresh
    grp['lat_qc'] = np.abs(robust_zscore(grp['lat_avg'])) < thresh
    grp['dur_qc'] = np.abs(robust_zscore(grp['dur_avg'])) < thresh

    if plot:
        plot_qc_thresh(grp['rtt_rt_med'], lb=-thresh, ub=thresh, title='RTT RT Median'   , ttype='zscore')
        plot_qc_thresh(grp['rew_tot'], lb=-thresh, ub=thresh, title='Total Reward'    , ttype='zscore')
        plot_qc_thresh(grp['dur_avg'], lb=-thresh, ub=thresh, title='Average Duration', ttype='zscore')
        plot_qc_thresh(grp['lat_avg'], lb=-thresh, ub=thresh, title='Average Latency' , ttype='zscore')

    # All PCA QCs
    grp = pca_quality_controls(grp, vars=['dur'], times=['avg'], plot=plot, thresh=thresh)

    # All foraging task QCs
    grp['for_qc'] = grp['lat_qc'] & grp['dur_qc'] & grp['rew_qc'] & grp['pca_qc']

    # Failure on anything, count
    grp['any_qc'] = grp[['exg_qc', 'for_qc']].all(axis=1)
    grp['cnt_qc'] = grp[['exg_qc', 'for_qc']].sum(axis=1)

    # Plot quality control failures
    if plot:
        qc_cols = ['dur_qc','rew_qc','pca_qc','lat_qc','rtt_qc','aes_qc','phq_qc','for_qc','exg_qc']
        plt.matshow(~grp[qc_cols], aspect='auto')
        ax = plt.gca()
        ax.set_xticks(np.arange(len(qc_cols)))
        ax.set_xticklabels(['dur','rew','pca','lat','rtt','aes','phq','for','exg'], rotation=45)
        plt.xlabel('Quality Control Check')
        plt.ylabel('Subject')
        plt.title('Quality Control Failures')

    # Drop foraging task outliers
    droplist = np.where(~grp['for_qc'])[0]
    if drop:
        subj, grp = drop_subjects(subj, grp, droplist, reset_index=True)
        n_dropped = len(droplist)
        print('Dropped', n_dropped, 'subjects for failing foraging task quality control.')
        print('Proportion subjects remaining:', 1.0 - n_dropped/np.shape(grp)[0])

    return subj, grp

def pca_quality_controls(grp, vars=['dur'], times=['avg'], thresh = 3.5, plot=False):
    '''Quality control checks for subject PCA scores data'''
    fails = np.zeros(len(grp)).astype(bool)
    for var, time in product(vars, times):
        for pc in ['pc1', 'pc2']:
            fld = var + '_' + time + '_'+ pc + '_score'
            grp[fld+'_qc'] = np.abs(robust_zscore(grp[fld])) < thresh

            if plot:
                plot_qc_thresh(grp[fld], lb=-thresh, ub=thresh, title=fld, ttype='zscore')

            fails += ~grp[fld+'_qc']

    grp['pca_qc'] = ~fails
    return grp

def plot_rtt_qc(grp, thresh=1500):
    x = nanless(grp['rtt_rt_med'].values)
    inds = np.argsort(x)
    zs = robust_zscore(x)[inds]
    outliers = x[inds] > thresh

    prctile = np.arange(0, 100, 100/len(zs))
    
    plt.figure(figsize=[3.3,3])
    plt.plot(prctile, zs          , '.')
    plt.plot(prctile[outliers], zs[outliers], 'r.')
    
    plt.xlabel('Subject Percentile')
    plt.ylabel('Robust Z-Score')
    plt.title('RTT RT Median QC')
    plt.grid()
    plt.tight_layout()

def plot_qc_thresh(x, lb, ub, title, ttype='value'):
    # Usage: x = grp[rew_tot].values
    x = nanless(x.values)
    inds = np.argsort(x)
    zs = robust_zscore(x)[inds]

    if ttype == 'value':
        outliers = (x[inds] < lb) | (x[inds] > ub)
    elif ttype == 'zscore':
        outliers = (zs < lb) | (zs > ub)

    prctile = np.arange(0, 100, 100/len(zs))
    
    plt.figure(figsize=[3.3*2,3])
    plt.subplot(1,2,1)

    plt.plot(prctile, zs          , '.')
    plt.plot(prctile[outliers], zs[outliers], 'r.')
    
    plt.xlabel('Subject Percentile')
    plt.ylabel('Robust Z-Score')
    plt.title(title + ' Z-Score')
    plt.grid()
    plt.tight_layout()

    plt.subplot(1,2,2)
    plt.plot(prctile, x[inds]          , '.')
    plt.plot(prctile[outliers], x[inds][outliers], 'r.')
    
    plt.xlabel('Subject Percentile')
    plt.ylabel('Value')
    plt.title(title + ' Value')
    plt.grid()
    plt.tight_layout()
