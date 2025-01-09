import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from analy import *
from utils import *

def plot_pc1_scores(grp):
    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['dur_avg_pc1_score'], grp['exit_avg_pc1_score'], c = grp.rew_tot, cmap = 'viridis', s=5)
    plt.xlabel('Duration pc1 Scores')
    plt.ylabel('Exit pc1 Scores')
    plt.tight_layout()

def plot_exit_pc_scores(grp):
    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['exit_pc1_score'], grp['exit_pc2_score'], cmap='viridis', c = grp.rew_tot)
    plt.xlabel('Exit pc1 Scores')
    plt.ylabel('Exit pc2 Scores')
    plt.tight_layout()

def plot_both_rve(grp):

    # Get policy models and associated info
    rve_dur , _ = get_reduced_model_rve(grp, type='dur')
    rve_exit, _ = get_reduced_model_rve(grp, type='exit')

    plt.figure(figsize = [3.3,3])
    plt.plot(rve_dur, '-om')
    plt.plot(rve_exit, '-or')
    plt.xlabel('PCs')
    plt.ylim([0,1])
    plt.legend(['Durations','Exits'])
    plt.ylabel('Cumulative Variance Explained')
    plt.tight_layout()



def plot_dur_devs(grp, by_age = False, show_title = False):
    
    devs = grp[['hl_dur_dev', 'll_dur_dev','hh_dur_dev', 'lh_dur_dev']].values/1000
    devs_opt = np.zeros(4)

    # Figure
    plt.figure(figsize = [3.3,3])

    plt.plot([0,1,2,3], devs_opt, '--ok')           # Optimal
    plt.plot([0,1,2,3], np.mean(devs,0) , '--or')   # Average over subjects

    # Subject data
    ns = np.shape(grp)[0]
    plt.scatter(np.random.normal(0,0.1,ns), devs[:,0], c = grp.rew_tot, cmap = 'viridis', s = 5)
    plt.scatter(np.random.normal(1,0.1,ns), devs[:,1], c = grp.rew_tot, cmap = 'viridis', s = 5)
    plt.scatter(np.random.normal(2,0.1,ns), devs[:,2], c = grp.rew_tot, cmap = 'viridis', s = 5)
    plt.scatter(np.random.normal(3,0.1,ns), devs[:,3], c = grp.rew_tot, cmap = 'viridis', s = 5)

    plt.legend(['Optimal', 'Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = plt.gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 30)
    plt.ylabel('Duration Deviations [s]')
    plt.tight_layout()


def plot_exit_threshold_pvalues(grp):

    plt.figure(figsize=(3.3,3))
    plt.hist(grp['exit_ftest_pval'], bins=np.arange(0,1,0.05), label='p-values')
    plt.xticks(np.arange(0,1,0.2))
    plt.ylabel('Counts')
    plt.xlabel('p-values')

    ylims = plt.ylim()
    plt.plot([0.05,0.05],ylims, 'r--', label='p = 0.05')
    plt.ylim(ylims)
    plt.legend()
    plt.tight_layout()

    # # Plot of p-values by pc1 scores
    # plt.figure(figsize=(3.3,3))
    # plt.plot(grp['dur_pc1_score'], np.log(grp['exit_ftest_pval']), 'o')
    # plt.xlabel('Dur. pc1 Scores')
    # plt.ylabel('log(p-value)')
    # plt.tight_layout()

    # # Plot of p-values by mean duration standard deviation
    # plt.figure(figsize=(3.3,3))
    # plt.plot(std_mean, np.log(grp['exit_ftest_pval']), 'o')
    # plt.xlabel('Mean Dur. Std')
    # plt.ylabel('log(p-value)')
    # plt.tight_layout()

    # Linear models
    # run_lm(np.log(grp['exit_ftest_pval']), grp['dur_pc1_score'], zscore=True)

    #std_mean = np.mean(grp[['hl_dur_std','hh_dur_std','ll_dur_std', 'lh_dur_std']], axis=1)
    #run_lm(np.log(grp['exit_ftest_pval']), std_mean, zscore=True)

    #run_lm(np.log(grp['exit_ftest_pval']), np.stack([std_mean, grp['dur_pc1_score']],axis=1), zscore=True)


def plot_stay_lms(subj, grp):

    results_marginal = fit_stay_lms_marginal(subj)
    results_full     = fit_stay_lms_full(subj)
    
    inds = np.argsort(results_marginal['r2']).values

    # Plot R values
    plt.figure(figsize=[3.3,3])
    plt.plot(results_marginal['r2'].values[inds], '.', zorder=10)
    plt.plot(results_full['r2'].values[inds], '.', zorder=11)
    plt.ylim([0,1])
    plt.grid()
    plt.xlabel('Subject Rank')
    plt.ylabel('R-squared')
    plt.tight_layout()

    # Plot log of model F-statistic p-values
    plt.figure(figsize=[3.3,3])
    plt.plot(np.log(results_marginal['fp'].values[inds]), '.', zorder=10)
    plt.plot(np.log(results_full['fp'].values[inds]), '.', zorder=11)
    plt.grid()
    plt.xlabel('Subject Rank')
    plt.ylabel('log(F-statistic p-value)')
    plt.tight_layout()

    plt.figure(figsize=[3.3,3])
    plt.plot(results_marginal['b_const'].values[inds], '.', zorder=10)
    plt.plot(results_full['b_const'].values[inds], '.', zorder=11)
    plt.grid()
    plt.xlabel('Subject Rank')
    plt.ylabel('Constant Coefficient')
    plt.tight_layout()

    # Plot model start coefficients
    plt.figure(figsize=[3.3,3])
    plt.plot(results_marginal['b_h-'].values[inds], '.', zorder=10)
    plt.plot(results_full['b_h-'].values[inds], '.', zorder=11)
    plt.grid()
    plt.xlabel('Subject Rank')
    plt.ylabel('Initial Reward Rate Coefficient')
    plt.tight_layout()

    # Plot model 
    plt.figure(figsize=[3.3,3])
    plt.plot(results_marginal['b_-h'].values[inds], '.', zorder=10)
    plt.plot(results_full['b_-h'].values[inds], '.', zorder=11)
    plt.grid()
    plt.xlabel('Subject Rank')
    plt.ylabel('Reward Decay Rate Coefficient')
    plt.tight_layout()

    plt.figure(figsize=[3.3,3])
    plt.plot(results_full['b_hl'].values[inds], '.', zorder=11)
    plt.grid()
    plt.xlabel('Subject Rank')
    plt.ylabel('High-Low Coefficient')
    plt.tight_layout()

    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['dur_pc1_score'], grp['b_const'], zorder=10, c=grp.rew_tot, s=5)
    plt.xlabel('Dur. pc1 Scores')
    plt.ylabel('Constant Coefficient')
    plt.grid()
    plt.tight_layout()

    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['dur_pc2_score'], grp['b_hl'], zorder=10, c=grp.rew_tot, s=5)
    plt.xlabel('Dur. pc2 Scores')
    plt.ylabel('High-Low Coefficient')
    plt.grid()
    plt.tight_layout()

    print(f'Mean marginal model R-squared: {np.mean(results_marginal["r2"]):0.2f}')
    print(f'Mean full model R-squared: {np.mean(results_full["r2"]):0.2f}')

    print(f'Mean marginal model constant coefficient: {np.mean(results_marginal["b_const"]):0.2f}')
    print(f'Mean marginal model -h coefficient: {np.mean(results_marginal["b_-h"]):0.2f}')
    print(f'Mean marginal model h- coefficient: {np.mean(results_marginal["b_h-"]):0.2f}')

    print(f'Mean full model constant coefficient: {np.mean(results_full["b_const"]):0.2f}')
    print(f'Mean full model -h coefficient: {np.mean(results_full["b_-h"]):0.2f}')

    print(f'Mean full model h- coefficient: {np.mean(results_full["b_h-"]):0.2f}')
    print(f'Mean full model hl coefficient: {np.mean(results_full["b_hl"]):0.2f}')

    corr = sp.stats.pearsonr(grp['b_const'], grp['dur_pc1_score'])
    print(f'Correlation between constant coefficient and pc1 score: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')

    corr = sp.stats.pearsonr(grp['b_hl'], grp['dur_pc2_score'])
    print(f'Correlation between hl coefficient and pc2 score: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')

def plot_exit_diffs_by_time(grp):
    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['exit_delta_h0'], grp['exit_delta_h1'], c=grp['rew_tot'], s=5)
    plt.xlim([0,4])
    plt.ylim([0,4])
    plt.yticks(range(0,5))
    plt.plot([0,4], [0,4], 'k--')
    plt.xlabel('Exit Delta at Start')
    plt.ylabel('Exit Delta at End')
    plt.tight_layout()

    corr = sp.stats.pearsonr(grp['exit_delta_h0'], grp['exit_delta_h1'])
    print(f'Correlation between exit delta at h0 and h1: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')


def plot_exit_thresh_by_reward(grp, fname=None):
    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['exit_means_avg'], grp['rew_tot'], c = grp['rew_tot'], cmap = 'viridis', s=5)
    plt.xlabel('Average Exit Threshold [1/s]')
    plt.ylabel('Total Reward')
    #cbar = plt.colorbar()
    #cbar.set_label('Total Reward', rotation=270, va='bottom')
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname, dpi=dpi)



def plot_pca_quality_controls(grp):
    '''Plots quality control checks for subject PCA scores data'''
    passes = grp['all_pca_qc']
    for var, time in product(['dur', 'exit', 'hgt'], ['avg', 't0', 't1']):
        for pc in ['pc1', 'pc2']:
            fld = var + '_' + time + '_'+ pc + '_score'
            
            plt.figure(figsize=[3.3,3])

            plt.plot(grp[fld][ passes], 'o')
            plt.plot(grp[fld][~passes], 'o')

            plt.xlabel('Subject')
            plt.ylabel(fld)
            plt.tight_layout()



def plot_dur_diffs(grp, show_title = False):
    
    # Get optimal and subject policies
    policy_opt = get_policy_opt()
    avg = get_policies(grp)
    
    # Change units
    avg = avg*1e-3
    
    #
    ns = np.shape(grp)[0]
    
    # Pool indistinguishable conditions
    pooled = np.stack([avg.iloc[:,0], 
                       (avg.iloc[:,1] + avg.iloc[:,2])/2,
                       avg.iloc[:,3]]).T
    
    # Differences between conditions
    diffs  = np.stack([
        pooled[:,0] - pooled[:,1],
        pooled[:,1] - pooled[:,2]]).T
    
    # Optimal differences
    opta = (policy_opt[1] + policy_opt[2])/2
    opts = np.array([policy_opt[0] - opta, opta - policy_opt[3]])
    opts = opts*1e-3
    
    # Group means
    means = np.mean(diffs,0)
    
    # KDE computation
    #support = np.arange(0, 14, 0.1)
    #kde1 = sp.stats.gaussian_kde(diffs[:,0])
    #kde2 = sp.stats.gaussian_kde(diffs[:,1])
    #kde3 = sp.stats.gaussian_kde(diffs[:,2])

    #plot(support, kde1(support))
    #plot(support, kde2(support))
    #plot(support, kde3(support))
    
    plt.figure(figsize=[3.3,3])

    plt.scatter(opts[0] , opts[1] , c = 'k', marker = 'x', zorder = 100)
    plt.scatter(means[0], means[1], c = 'r', marker = 'x', zorder = 101)

    plt.scatter(diffs[:,0], diffs[:,1], c = grp.rew_tot)

    plt.xlabel('Best minus Mid Durations [s]')
    plt.ylabel('Mid minus Worst Durations [s]')
    plt.legend(['Optimal', 'Mean'])

    if show_title:
        plt.title('Condition Differences')

    plt.xlim([0,10])
    plt.ylim([0, 5])
    plt.tight_layout()

    corr = sp.stats.pearsonr(diffs[:,0], diffs[:,1])
    print(f'Correlation between best-mid and mid-worst duration differences: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')


def plot_latencies(grp, by_age = False, show_title = False):
    
    # Get optimal and subject policies
    avg = grp[ ['hl_lat_avg', 'll_lat_avg', 'hh_lat_avg', 'lh_lat_avg'] ]

    # Number of subjects
    ns = np.shape(grp)[0]
    
    # Color subjects by optimality
    cmap = 'viridis'
    
    # Change units to [s]
    avg = avg*1e-3

    # Figure
    plt.figure(figsize = [3.3,3])

    plt.plot([0,1,2,3], np.mean(avg,0) , '--or')

    # Subject data
    plt.scatter(np.random.normal(0,0.1,ns), avg.iloc[:,0], c = grp.rew_tot, cmap = cmap, s = 5)
    plt.scatter(np.random.normal(1,0.1,ns), avg.iloc[:,1], c = grp.rew_tot, cmap = cmap, s = 5)
    plt.scatter(np.random.normal(2,0.1,ns), avg.iloc[:,2], c = grp.rew_tot, cmap = cmap, s = 5)
    plt.scatter(np.random.normal(3,0.1,ns), avg.iloc[:,3], c = grp.rew_tot, cmap = cmap, s = 5)

    plt.legend(['Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = plt.gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 30)
    
    plt.ylabel('Response Latencies [s]')

    #ylim([0,20])
    if show_title:
        plt.title('Subject Response Latencies')
    plt.tight_layout()



def plot_exit_diffs(grp, show_title = False):
    
    
    # Get optimal and subject thresholds
    policy_opt = get_policy_opt()
    thresh_opt = get_policy_exit_thresholds(policy_opt)*1e3

    policy_avg = get_policies(grp)
    thresh_avg = get_policy_exit_thresholds(policy_avg)*1e3

    # Pool indistinguishable conditions
    pooled = np.stack([thresh_avg[:,0], 
                       (thresh_avg[:,1] + thresh_avg[:,2])/2,
                       thresh_avg[:,3]]).T
    
    # Differences between conditions
    diffs  = np.stack([
        pooled[:,0] - pooled[:,1],
        pooled[:,1] - pooled[:,2]]).T
    
    # Optimal differences
    opta = (thresh_opt[1] + thresh_opt[2])/2
    opts = np.array([thresh_opt[0] - opta, opta - thresh_opt[3]])
    
    # Group means
    means = np.mean(diffs,0)
    
    # KDE computation
    support = np.arange(0, 10, 0.01)
    kde1 = sp.stats.gaussian_kde(thresh_avg[:,0])
    kde2 = sp.stats.gaussian_kde(thresh_avg[:,1])
    kde3 = sp.stats.gaussian_kde(thresh_avg[:,2])
    kde4 = sp.stats.gaussian_kde(thresh_avg[:,3])

    plt.figure(figsize=[3.3,3])
    plt.plot(support, kde1(support))
    plt.plot(support, kde2(support))
    plt.plot(support, kde3(support))
    plt.plot(support, kde4(support))
    
    plt.figure(figsize=[3.3,3])

    plt.scatter(opts[0] , opts[1] , c = 'k', marker = 'x', zorder = 100)
    plt.scatter(means[0], means[1], c = 'r', marker = 'x', zorder = 101)

    plt.scatter(diffs[:,0], diffs[:,1], c = grp.rew_tot)

    plt.xlabel('Best minus Mid Thresholds [L/s]')
    plt.ylabel('Mid minus Worst Thresholds [L/s]')
    plt.legend(['Optimal', 'Mean'])

    if show_title:
        plt.title('Condition Differences')

    #xlim([0,10])
    #ylim([0, 5])
    plt.tight_layout()
    
    


def plot_reward_dist(grp, show_title = False):
    # Get reward KDEs
    support = np.arange(10, 30, 0.2)
    kde = sp.stats.gaussian_kde(grp.rew_tot)

    # Reward under optimal policy
    duration = get_policy_opt()
    lat_meds = grp[ ['hl_lat_avg', 'll_lat_avg', 'hh_lat_avg', 'lh_lat_avg'] ].median().to_list()
    reward   = get_policy_rew(duration, latency = lat_meds)

    # Plot
    plt.figure(figsize = [3.3,3])
    plt.plot(support, kde(support))
    ylims = plt.ylim()
    plt.plot([reward, reward], ylims, '--k')
    plt.ylim(ylims)

    plt.xlabel('Total Reward')
    plt.ylabel('Density')
    plt.legend(['Subjects', 'Optimal'])
    if show_title:
        plt.title('Total Reward Distribution')
    plt.tight_layout()
