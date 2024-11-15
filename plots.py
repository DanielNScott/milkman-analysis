# External imports
import numpy as np
import scipy as sp
import scipy.stats as st


from matplotlib.pyplot import figure, plot, imshow, subplot, errorbar, gca, scatter
from matplotlib.pyplot import title, xlabel, ylabel, xlim, ylim, grid, tight_layout, close, legend, yticks, xticks
from matplotlib.pyplot import ioff, ion
from matplotlib import pyplot as plt

from datetime          import datetime
from os                import mkdir

from stats import *

# Imports from my modules
from analy import *

def plot_conditions():
    t = np.arange(0,20000)
    conds = ((0.02, 0.0002), (0.01, 0.0002), (0.02, 0.0004), (0.01, 0.0004))

    dmdt, rews = [],[]
    for cond in conds:
        dmdt.append(cond[0]* np.exp(-cond[1]*t))
        rews.append(1/100 * cond[0] / cond[1] * (1 - np.exp(-cond[1] * t)))

    # Reward rate curves
    plt.figure(figsize=[3.3,3])
    for i in range(4):
        plt.plot(t/1000, dmdt[i]*1000)

    plt.xlabel('Time [s]')
    plt.ylabel('Reward Rate [1/ms]')
    plt.legend(['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast'])
    plt.tight_layout()

    # Accumulated reward curves
    plt.figure(figsize=[3.3,3])
    for i in range(4):
        plt.plot(t/1000, rews[i])

    plt.xlabel('Time [s]')
    plt.ylabel('Total Reward')
    plt.legend(['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast'])
    plt.tight_layout()


def plot_durations(grp, by_age = False, show_title = False):
    
    # Get optimal and subject policies
    policy_opt = get_policy_opt()*1e-3
    avg = get_policies(grp)*1e-3

    # Number of subjects
    ns = np.shape(grp)[0]
    
    # Color subjects by optimality
    cmap = 'viridis'

    # Figure
    figure(figsize = [3.3,3])

    # Optimal policy * group averages
    plot([0,1,2,3], policy_opt, '--ok')

    if by_age:
        avg_young = np.mean(avg[grp['age_cat'] == 0])
        avg_old   = np.mean(avg[grp['age_cat'] == 1])
        plot([0,1,2,3], avg_young , '--om')
        plot([0,1,2,3], avg_old   , '--or')
    else:
        plot([0,1,2,3], np.mean(avg,0) , '--or')

    # Subject data
    scatter(np.random.normal(0,0.1,ns), avg.iloc[:,0], c = grp.rew_tot, cmap = cmap, s = 5)
    scatter(np.random.normal(1,0.1,ns), avg.iloc[:,1], c = grp.rew_tot, cmap = cmap, s = 5)
    scatter(np.random.normal(2,0.1,ns), avg.iloc[:,2], c = grp.rew_tot, cmap = cmap, s = 5)
    scatter(np.random.normal(3,0.1,ns), avg.iloc[:,3], c = grp.rew_tot, cmap = cmap, s = 5)

    if by_age:
        legend(['Optimal', 'Means (young)', 'Means (old)'])
    else:
        legend(['Optimal', 'Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 30)
    
    ylabel('Stay Durations [s]')

    ylim([0,20])
    if show_title:
        title('Subject Stay-Durations')
    tight_layout()


    formatted_policy_opt = ' '.join([f'{x:0.2e}' for x in policy_opt])
    print(f'Optimal durations: {formatted_policy_opt}')

    formatted_policy_avg = ' '.join([f'{x:0.2e}' for x in np.mean(avg,0)])
    print(f'Average durations: {formatted_policy_avg}')

    formatted_diffs = ' '.join([f'{x:0.2e}' for x in np.mean(avg,0) - policy_opt])
    print(f'Differences: {formatted_diffs}')

    compare_means(avg.iloc[:,0], avg.iloc[:,1], labels=['High-Slow', 'Low-Slow'])
    compare_means(avg.iloc[:,1], avg.iloc[:,2], labels=['Low-Slow' , 'High-Fast'])
    compare_means(avg.iloc[:,2], avg.iloc[:,3], labels=['High-Fast', 'Low-Fast'])

    compare_means(avg.iloc[:,0], policy_opt[[0]], labels=['High-Slow', 'Optimal'])
    compare_means(avg.iloc[:,1], policy_opt[[1]], labels=['Low-Slow', 'Optimal'])
    compare_means(avg.iloc[:,2], policy_opt[[2]], labels=['High-Fast', 'Optimal'])
    compare_means(avg.iloc[:,3], policy_opt[[3]], labels=['Low-Fast', 'Optimal'])

    corr = sp.stats.pearsonr(-grp['dur_avg'], grp['rew_tot'])
    print(f'Correlation between average duration and total reward: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')


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
    
    figure(figsize=[3.3,3])

    scatter(opts[0] , opts[1] , c = 'k', marker = 'x', zorder = 100)
    scatter(means[0], means[1], c = 'r', marker = 'x', zorder = 101)

    scatter(diffs[:,0], diffs[:,1], c = grp.rew_tot)

    xlabel('Best minus Mid Durations [s]')
    ylabel('Mid minus Worst Durations [s]')
    legend(['Optimal', 'Mean'])

    if show_title:
        title('Condition Differences')

    xlim([0,10])
    ylim([0, 5])
    tight_layout()

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
    figure(figsize = [3.3,3])

    plot([0,1,2,3], np.mean(avg,0) , '--or')

    # Subject data
    scatter(np.random.normal(0,0.1,ns), avg.iloc[:,0], c = grp.rew_tot, cmap = cmap, s = 5)
    scatter(np.random.normal(1,0.1,ns), avg.iloc[:,1], c = grp.rew_tot, cmap = cmap, s = 5)
    scatter(np.random.normal(2,0.1,ns), avg.iloc[:,2], c = grp.rew_tot, cmap = cmap, s = 5)
    scatter(np.random.normal(3,0.1,ns), avg.iloc[:,3], c = grp.rew_tot, cmap = cmap, s = 5)

    legend(['Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 30)
    
    ylabel('Response Latencies [s]')

    #ylim([0,20])
    if show_title:
        title('Subject Response Latencies')
    tight_layout()


def plot_exits(grp, by_age = False, show_title = False):
    
    # Get optimal and subject policies
    policy_opt = get_policy_opt()
    thresh_opt = get_policy_exit_thresholds(policy_opt)*1e3

    policy_avg = get_policies(grp)
    avg = get_policy_exit_thresholds(policy_avg)

    # Number of subjects
    ns = np.shape(grp)[0]
    
    # Color subjects by optimality
    cmap = 'viridis'
    
    # Change units to [s]
    avg = avg*1e3

    # Figure
    figure(figsize = [3.3,3])

    # Optimal policy * group averages
    plot([0,1,2,3], thresh_opt, '--ok')

    plot([0,1,2,3], np.mean(avg,0) , '--or')

    # Subject data
    scatter(np.random.normal(0,0.1,ns), avg[:,0], c = grp.rew_tot, cmap = cmap, s = 5)
    scatter(np.random.normal(1,0.1,ns), avg[:,1], c = grp.rew_tot, cmap = cmap, s = 5)
    scatter(np.random.normal(2,0.1,ns), avg[:,2], c = grp.rew_tot, cmap = cmap, s = 5)
    scatter(np.random.normal(3,0.1,ns), avg[:,3], c = grp.rew_tot, cmap = cmap, s = 5)

    legend(['Optimal', 'Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 30)
    
    ylabel('Exit Thresholds [1/s]')

    #ylim([0,20])
    if show_title:
        title('Subject Exit Thresholds')
    tight_layout()

    formatted_thresh_opt = ' '.join([f'{x:0.2e}' for x in thresh_opt])
    print(f'Optimal thresholds: {formatted_thresh_opt}')

    formatted_thresh_avg = ' '.join([f'{x:0.2e}' for x in np.mean(avg,0)])
    print(f'Average thresholds: {formatted_thresh_avg}')
    
    compare_means(avg[:,0], avg[:,1], labels=['High-Slow', 'Low-Slow'])
    compare_means(avg[:,1], avg[:,2], labels=['Low-Slow' , 'High-Fast'])
    compare_means(avg[:,2], avg[:,3], labels=['High-Fast', 'Low-Fast'])

    compare_means(avg[:,0], thresh_opt[[0]], labels=['High-Slow', 'Optimal'])
    compare_means(avg[:,1], thresh_opt[[1]], labels=['Low-Slow', 'Optimal'])
    compare_means(avg[:,2], thresh_opt[[2]], labels=['High-Fast', 'Optimal'])
    compare_means(avg[:,3], thresh_opt[[3]], labels=['Low-Fast', 'Optimal'])

    corr = sp.stats.pearsonr(-grp['dur_avg'], grp['rew_tot'])
    print(f'Correlation between average exit and total reward: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')



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

    figure(figsize=[3.3,3])
    plot(support, kde1(support))
    plot(support, kde2(support))
    plot(support, kde3(support))
    plot(support, kde4(support))
    
    figure(figsize=[3.3,3])

    scatter(opts[0] , opts[1] , c = 'k', marker = 'x', zorder = 100)
    scatter(means[0], means[1], c = 'r', marker = 'x', zorder = 101)

    scatter(diffs[:,0], diffs[:,1], c = grp.rew_tot)

    xlabel('Best minus Mid Thresholds [L/s]')
    ylabel('Mid minus Worst Thresholds [L/s]')
    legend(['Optimal', 'Mean'])

    if show_title:
        title('Condition Differences')

    #xlim([0,10])
    #ylim([0, 5])
    tight_layout()
    
    


def plot_reward_dist(grp, show_title = False):
    # Get reward KDEs
    support = np.arange(10, 30, 0.2)
    kde = sp.stats.gaussian_kde(grp.rew_tot)

    # Reward under optimal policy
    duration = get_policy_opt()
    lat_meds = grp[ ['hl_lat_avg', 'll_lat_avg', 'hh_lat_avg', 'lh_lat_avg'] ].median().to_list()
    reward   = get_policy_rew(duration, latency = lat_meds)

    # Plot
    figure(figsize = [3.3,3])
    plot(support, kde(support))
    ylims = plt.ylim()
    plot([reward, reward], ylims, '--k')
    ylim(ylims)

    xlabel('Total Reward')
    ylabel('Density')
    legend(['Subjects', 'Optimal'])
    if show_title:
        title('Total Reward Distribution')
    tight_layout()


def plot_stay_pca(summary, type = 'dur', time = 'avg'):
    keys = ['_'.join([type,time,'pc'+str(i)]) for i in range(1,4)]
    pcs =   np.stack([summary[key       ] for key in keys])
    err = 2*np.stack([summary[key+'_jse'] for key in keys])

    # Plot of PCs
    plot_grouped_bars(pcs[0:2,:], xticklabels=['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast'], ylabel='Loadings', rotation = 30, figsize=[3.3,3], yerr=err)
    plt.legend(['PC1','PC2'],loc='lower left')

    # Plot of variance explained
    #plt.figure(figsize=[3.3,3])
    #plt.plot(np.cumsum(evals)/sum(evals), '-o', zorder=10)
    #plt.xlabel('Principal Component')
    #plt.ylabel('Cumulative Variance Explained')
    #plt.tight_layout()

    #corr = sp.stats.pearsonr(-grp['dur_pc1_score'], grp['rew_tot'])
    #print(f'Correlation between pc1 score and total reward: {corr.statistic:0.2e}')
    #print(f'p-value: {corr.pvalue:0.2e}')

def plot_pc1_scores(grp):
    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['dur_avg_pc1_score'], grp['exit_avg_pc1_score'], c = grp.rew_tot, cmap = 'viridis', s=5)
    plt.xlabel('Duration pc1 Scores')
    plt.ylabel('Exit pc1 Scores')
    plt.tight_layout()




def plot_both_ve(grp):
    durs = np.array(grp[ ['hl_dur_avg', 'll_dur_avg', 'hh_dur_avg', 'lh_dur_avg'] ])
    evals, evecs = np.linalg.eig(np.cov(durs.T))

    if evecs[0,0] < 0:
        evecs[:,0] = -evecs[:,0]

    if evecs[0,1] < 0:
        evecs[:,1] = -evecs[:,1]

    if evecs[2,2] < 0:
        evecs[:,2] = -evecs[:,2]

    evals_dur = evals

    exits = np.array(grp[ ['hl_exit_avg', 'll_exit_avg', 'hh_exit_avg', 'lh_exit_avg'] ])
    evals, evecs = np.linalg.eig(np.cov(exits.T))

    if evecs[0,0] < 0:
        evecs[:,0] = -evecs[:,0]

    if evecs[0,1] < 0:
        evecs[:,1] = -evecs[:,1]

    if evecs[2,2] < 0:
        evecs[:,2] = -evecs[:,2]

    # Plot of variance explained
    plt.figure(figsize=[3.3,3])
    plt.plot([1,2,3,4], np.cumsum(evals_dur)/sum(evals_dur), '-om', zorder=10, label='Stay-Duration')
    plt.plot([1,2,3,4], np.cumsum(evals)/sum(evals), '-or', zorder=10, label='Exit-Threshold')
    plt.legend()

    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Proportion VE')
    plt.tight_layout()


def plot_dur_pc_scores(grp):
    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['dur_pc1_score'], grp['dur_pc2_score'], cmap='viridis', c = grp.rew_tot)
    plt.xlabel('Dur. pc1 Scores')
    plt.ylabel('Dur. pc2 Scores')
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

    figure(figsize = [3.3,3])
    plt.plot(rve_dur, '-om')
    plt.plot(rve_exit, '-or')
    plt.xlabel('PCs')
    plt.ylim([0,1])
    legend(['Durations','Exits'])
    ylabel('Cumulative Variance Explained')
    tight_layout()


def plot_exit_devs(grp, by_age = False, show_title = False, sfx=''):
    
    devs = grp[['hl_exit_dev'+sfx, 'll_exit_dev'+sfx,'hh_exit_dev'+sfx, 'lh_exit_dev'+sfx]].values
    devs_opt = np.zeros(4)

    # Figure
    figure(figsize = [3.3,3])

    plot([0,1,2,3], devs_opt, '--ok')           # Optimal
    plot([0,1,2,3], np.mean(devs,0) , '--or')   # Average over subjects

    # Subject data
    ns = np.shape(grp)[0]
    scatter(np.random.normal(0,0.1,ns), devs[:,0], c = grp.rew_tot, cmap = 'viridis', s = 5)
    scatter(np.random.normal(1,0.1,ns), devs[:,1], c = grp.rew_tot, cmap = 'viridis', s = 5)
    scatter(np.random.normal(2,0.1,ns), devs[:,2], c = grp.rew_tot, cmap = 'viridis', s = 5)
    scatter(np.random.normal(3,0.1,ns), devs[:,3], c = grp.rew_tot, cmap = 'viridis', s = 5)

    legend(['Optimal', 'Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 30)
    ylabel('Exit Threshold Dev.s')
    tight_layout()

    compare_means(avg.iloc[:,0], avg.iloc[:,1], labels=['High-Slow', 'Low-Slow'])
    compare_means(avg.iloc[:,1], avg.iloc[:,2], labels=['Low-Slow' , 'High-Fast'])
    compare_means(avg.iloc[:,2], avg.iloc[:,3], labels=['High-Fast', 'Low-Fast'])

    compare_means(avg.iloc[:,0], policy_opt[[0]], labels=['High-Slow', 'Optimal'])
    compare_means(avg.iloc[:,1], policy_opt[[1]], labels=['Low-Slow', 'Optimal'])
    compare_means(avg.iloc[:,2], policy_opt[[2]], labels=['High-Fast', 'Optimal'])
    compare_means(avg.iloc[:,3], policy_opt[[3]], labels=['Low-Fast', 'Optimal'])

    corr = sp.stats.pearsonr(-grp['exit_delta'], grp['rew_tot'])
    print(f'Correlation between exit delta and total reward: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')



    corr = sp.stats.pearsonr(-grp['exit_pc1_score'], grp['rew_tot'])
    print(f'Correlation between pc1 score and total reward: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')





def plot_dur_devs(grp, by_age = False, show_title = False):
    
    devs = grp[['hl_dur_dev', 'll_dur_dev','hh_dur_dev', 'lh_dur_dev']].values/1000
    devs_opt = np.zeros(4)

    # Figure
    figure(figsize = [3.3,3])

    plot([0,1,2,3], devs_opt, '--ok')           # Optimal
    plot([0,1,2,3], np.mean(devs,0) , '--or')   # Average over subjects

    # Subject data
    ns = np.shape(grp)[0]
    scatter(np.random.normal(0,0.1,ns), devs[:,0], c = grp.rew_tot, cmap = 'viridis', s = 5)
    scatter(np.random.normal(1,0.1,ns), devs[:,1], c = grp.rew_tot, cmap = 'viridis', s = 5)
    scatter(np.random.normal(2,0.1,ns), devs[:,2], c = grp.rew_tot, cmap = 'viridis', s = 5)
    scatter(np.random.normal(3,0.1,ns), devs[:,3], c = grp.rew_tot, cmap = 'viridis', s = 5)

    legend(['Optimal', 'Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 30)
    ylabel('Duration Deviations [s]')
    tight_layout()


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

def plot_pcs_by_time(summary):
    # Stack PCs
    colors = {'pc1':['#336ea0', '#679fce', '#b1cee6'], 'pc2':['#d95f02', '#f4a582', '#fddbc7']}


    for type in ['dur', 'exit']:
        for pc in ['pc1', 'pc2']:
            pcs = np.stack([summary[key] for key in [type+'_avg_'+pc, type+'_t0_'+pc, type+'_t1_'+pc]])
            err = np.stack([summary[key] for key in [type+'_avg_'+pc+'_jse', type+'_t0_'+pc+'_jse', type+'_t1_'+pc+'_jse']])

            plot_grouped_bars(pcs, xticklabels=['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast'], ylabel=pc+' Loadings', rotation = 30, figsize=[3.3,3], yerr=2*err, colors=colors[pc])
            plt.legend(['Average','Start','End'])


            normed_t0 = summary[type+'_t0_'+pc]/np.linalg.norm(summary[type+'_t0_'+pc])
            normed_t1 = summary[type+'_t1_'+pc]/np.linalg.norm(summary[type+'_t1_'+pc])

            angle = np.arccos(np.dot(normed_t0, normed_t1))
            print(f'Angle between {type} {pc} scores at t0 and t1: {angle*360/(2*np.pi):0.2f}')

            #corr = sp.stats.pearsonr(summary[type+'_t0_'+pc], summary[type+'_t1_'+pc])
            #print(f'Correlation between {type} {pc} score at t0 and {pc} score at t1: {corr.statistic:0.2f}')
            #print(f'p-value: {corr.pvalue:0.2e}')


def plot_pc_scores_by_time(grp):
    for var, varname in {'dur':'Duration', 'exit':'Exit', 'hgt':'Height'}.items():
        for pc, pcname in {'pc1':'PC1', 'pc2':'PC2'}.items():
            plt.figure(figsize=[3.3,3])
            plt.scatter(grp[var+'_h0_'+pc+'_score'], grp[var+'_h1_'+pc+'_score'], c = grp.rew_tot, cmap = 'viridis', s=5)
            plt.xlabel(f'{varname} Start {pcname} Scores')
            plt.ylabel(f'{varname} End {pcname} Scores')
            if pc == 'pc1':
                #plt.xlim([-1,2.5])
                #plt.ylim([-1,2.5])
                #plt.plot([-1,2.5], [-1,2.5], 'k--')
                pass

            if pc == 'pc2':
                #plt.xlim([-1,2.5])
                #plt.ylim([-1,2.5])
                #plt.plot([-1,2.5], [-1,2.5], 'k--')
                pass

            plt.tight_layout()

            #good = grp[key+'_t0_pc2_score']>-1.0
            corr = sp.stats.spearmanr(grp[f'{var}_h0_'+pc+'_score'], grp[f'{key}_h1_'+pc+'_score'])
            print(f'Correlation between {var} {pc} score at h0 and {pc} score at h1: {corr.statistic:0.2f}')
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

def plot_exit_dev_corrs_by_time(grp):
    for var, varname in {'exit':'Exit'}.items():

        conds = ['hh', 'hl', 'lh', 'll']
        pre = ['_'.join([c, var, 'h0', 'dev']) for c in conds]
        pst = ['_'.join([c, var, 'h1', 'dev']) for c in conds]
        

        corrs, pvals, ub, lb, delta = [], [], [], [], 1.96/np.sqrt(len(grp)-3)
        for h0, h1 in zip(pre, pst):
            # Correlation
            corrs.append(sp.stats.spearmanr(grp[h0], grp[h1]).statistic)
            pvals.append(sp.stats.spearmanr(grp[h0], grp[h1]).pvalue)

            # Use Fisher transform of Spearman rank correlation to approximate z-score
            ub.append(np.tanh(np.arctanh(corrs[-1]) + delta))
            lb.append(np.tanh(np.arctanh(corrs[-1]) - delta))

        corrs, ub, lb, pvals = np.array(corrs), np.array(ub), np.array(lb), np.array(pvals)
        ses = np.mean(np.stack([ub-corrs,corrs-lb]),axis=0)

        xticklabels = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']

        fig, ax = plt.subplots(figsize=[3.3,3])
        ax.bar(x=range(4), height=corrs, yerr=ses, color='#826ad3')
        ax.set_xticks(range(4))
        ax.set_xticklabels(xticklabels, rotation=30)
        ax.set_ylabel('Spearman Rank Correlation')
        plt.tight_layout()

        print(f'Deviation correlations {varname}: {corrs}')
        print(f'Deviation p-values {varname}: {pvals}')

        print(f'')
        exit_corr = sp.stats.spearmanr(grp['exit_delta_avg'], grp['exit_avg_pc1_score'])


def plot_exit_devs_by_time(summary):
    # Stack PCs
    colors = ['#4d31aa', '#826ad3', '#c2b6e9']

    pcs = np.stack([summary[key] for key in ['exit_dev_means_avg', 'exit_dev_means_t0', 'exit_dev_means_t1']])
    err = np.stack([summary[key] for key in ['exit_dev_sems_avg', 'exit_dev_sems_t0', 'exit_dev_sems_t1']])

    plot_grouped_bars(pcs, xticklabels=['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast'], ylabel='Average', rotation = 30, figsize=[3.3,3], yerr=2*err, colors=colors)
    plt.legend(['Average','Start','End'])

    #normed_t0 = summary['exit_dev_means_t0_'+pc]/np.linalg.norm(summary['exit_dev_means_t0_'+pc])
    #normed_t1 = summary['exit_dev_means_t1_'+pc]/np.linalg.norm(summary['exit_dev_means_t1_'+pc])

    #angle = np.arccos(np.dot(normed_t0, normed_t1))
    #print(f'Angle between {type} {pc} scores at t0 and t1: {angle*360/(2*np.pi):0.2f}')

    #corr = sp.stats.pearsonr(summary['exit_dev_means_t0], summary['exit_dev_means_t1])
    #print(f'Correlation between {type} {pc} score at t0 and {pc} score at t1: {corr.statistic:0.2f}')
    #print(f'p-value: {corr.pvalue:0.2e}')


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



def plot_policy_prediction_lms(grp, residualize = True):

    # Regressors for primary outcome variables, excluding age
    #exog_cols = [
    #    'aes_score'   , 'phq_score'   , 'tol_acc',
    #    'aes_rt_med'  , 'phq_rt_med'  , 'tol_rtc_med'  , 'tol_rti_med'  , 'rtt_rt_med',
    #    'aes_rt_mad'  , 'phq_rt_mad'  , 'tol_rtc_mad'  , 'tol_rti_mad'  , 'rtt_rt_mad',
    #    'aes_rt_lapse', 'phq_rt_lapse', 'tol_rtc_lapse', 'tol_rti_lapse', 'rtt_rt_lapse',
    #    'lat_med']

    exog_cols = ['aes_score', 'phq_score', 'tol_acc', 'tol_rtc_med', 'rtt_rt_med']

    # Things we try to predict
    endog_cols  = ['rew_tot', 'dur_avg_pc1_score', 'dur_avg_pc2_score', 'll_exit_avg_dev']
    endog_names = ['Total Reward', 'Duration PC1', 'Duration PC2', 'Low-Slow Exit Deviation']

    # Predictor / design matrix (without constant)
    exog = grp[exog_cols].loc[grp['all_exog_qc']].copy()
    age  = grp[['age']].loc[grp['all_exog_qc']].copy()

    # Find any locations where there are NaNs
    no_exog = np.where(exog.isna().any(axis=1))[0]
    no_age  = np.where(age.isna().any(axis=1))[0]
    keep    = np.setdiff1d(np.arange(len(exog)), np.union1d(no_exog, no_age))

    # Subset data
    exog = exog.iloc[keep].reset_index(drop=True)
    age  = age.iloc[keep].reset_index(drop=True)

    # Get relationships to age and residualize
    resid, age_models = run_age_lms(exog, age)

    # Run linear model predicting endogenous from residuals and age
    endog     = grp[endog_cols].copy().iloc[keep].reset_index(drop=True) 
    models    = run_lms(endog, age.join(resid))
    regcnames = ['age'] + exog_cols

    # Age model results
    plot_age_lm_results(age_models, figsize=[3.3*1,3])

    #
    #plot_correlation_matrix(grp  , regcnames)
    #plot_correlation_matrix(resid, regcnames)

    #
    #plot_lm_results(models, cols= ['rew_tot', 'lat_med'], cnames = regcnames, labels = ['Total Reward', 'Median Latency'])
    #plot_lm_results(models, cols= ['b_h-', 'b_-h'], cnames = regcnames, labels = ['Initial Reward Rate', 'Reward Decay Rate'])
    #plot_lm_results(models, cols= ['baseline_composite', 'hl_composite'], cnames = regcnames, labels = ['Overstaying', 'High-Slow Preference'])
    #plot_lm_results(models, cols= ['ll_dur_dev', 'hh_dur_dev'], cnames = regcnames, labels = ['Low-Slow Deviation', 'High-Fast Deviation'])
    #plot_lm_results(models, cols= ['exit_delta'], cnames = regcnames, labels = ['Exit Delta'])

    for endog, name in zip(endog_cols, endog_names):
        plot_lm_results(models, cols= [endog], cnames = regcnames, labels = [name], figsize=[3.3*1,3])


    # Predict PC1 scores from age and residuals
    res = run_lm(endog['dur_avg_pc1_score'], age.join(resid), zscore=True)
    preds = res.predict(age.join(resid))
    rew_tot = grp['rew_tot'].copy().iloc[keep].reset_index(drop=True) 
    plt.figure(figsize=[3.3,3])
    plt.scatter(endog['dur_avg_pc1_score'], preds, c = rew_tot, cmap = 'viridis', s=5)
    plt.xlabel('True PC1 Scores')
    plt.ylabel('Predicted PC1 Scores')
    plt.tight_layout()

    x = grp['age']
    y = grp['dur_avg_pc1_score']
    scatter_with_lm_fit(x=x, y=y, c=grp['rew_tot'], xlabel='Age', ylabel='PC1 Score')

    x = grp['tol_rtc_med']/1000
    y = grp['dur_avg_pc1_score']
    scatter_with_lm_fit(x=x, y=y, c=grp['rew_tot'], xlabel='ToL RTc Median [s]', ylabel='PC1 Score')

    x = resid['phq_score']
    y = endog['dur_avg_pc1_score']
    scatter_with_lm_fit(x=x, y=y, c=rew_tot, xlabel='Residualised PHQ Score', ylabel='PC1 Score')

    x = grp['phq_score']
    y = grp['dur_avg_pc1_score']
    scatter_with_lm_fit(x=x, y=y, c=grp['rew_tot'], xlabel='Raw PHQ Score', ylabel='PC1 Score')

    x = resid['tol_rtc_med']
    y = endog['dur_avg_pc1_score']
    scatter_with_lm_fit(x=x, y=y, c=rew_tot, xlabel='Residualised ToL RTc Median', ylabel='PC1 Score')

    return

    #corr = sp.stats.pearsonr(grp['aes_rt_med'], grp['phq_rt_med'])

    # Separate affective, cognitive, and speed aspects
    affective = ['aes_score', 'phq_score', 'aes_rt_med', 'phq_rt_med', 'aes_rt_mad', 'phq_rt_mad', 'aes_rt_lapse', 'phq_rt_lapse']
    cognitive = ['tol_acc', 'tol_rtc_med', 'tol_rti_med', 'tol_rtc_mad', 'tol_rti_mad', 'tol_rtc_lapse', 'tol_rti_lapse']
    speed     = ['rtt_rt_med', 'rtt_rt_mad','rtt_rt_lapse']

    plot_correlation_matrix(resid[affective], affective, figsize=[3.3*1.75, 3.3*1.5])
    plot_correlation_matrix(resid[cognitive], cognitive, figsize=[3.3*1.75, 3.3*1.5])
    plot_correlation_matrix(resid[speed    ], speed,     figsize=[3.3*1.25, 3.3*1.0])

    # Separate affective score vs timing info
    ascores = ['aes_score', 'phq_score']
    atimes  = ['aes_rt_med', 'phq_rt_med', 'aes_rt_mad', 'phq_rt_mad']
    alapse  = ['aes_rt_lapse', 'phq_rt_lapse']

    # Get new affective basis
    evals, evecs, scores = PCA(resid[ascores], pos_const = True)
    affect_df = pd.DataFrame(scores, columns = ['AES+PHQ', 'PHQ-AES'])

    # Get affective time basis
    evals, evecs, scores = PCA(resid[atimes])
    atimes_pc_cols = ['A++++', 'A-+-+', 'A--++', 'A-++-']
    atimes_df = pd.DataFrame(scores, columns = atimes_pc_cols)

    plot_correlation_matrix(resid[atimes], atimes, figsize=[3.3*1.75, 3.3*1.5])
    plot_grouped_bars(evecs.T, xticklabels=atimes, ylabel='Loadings', rotation = 30, figsize=[3.3*3,3])
    plot_scree_cumsum(evals, pc_names = atimes_pc_cols)

    # Get 
    evals, evecs, scores = PCA(resid[cognitive])
    cog_pc_cols = ['CogAccLowRTILapse', 'CogSlow', 'CogAccLowRTIMad', 'CogFastRTI', 'CogAccHighRTIMad']
    cog_df = pd.DataFrame(scores[:,0:5], columns = cog_pc_cols)

    plot_correlation_matrix(resid[cognitive], cognitive, figsize=[3.3*1.75, 3.3*1.5])
    plot_grouped_bars(evecs.T, xticklabels=cognitive, ylabel='Loadings', rotation = 30, figsize=[3.3*3,3])
    plot_scree_cumsum(evals, pc_names = cog_pc_cols + ['PC6', 'PC7'])

    # Get 
    evals, evecs, scores = PCA(resid[speed])
    speed_pc_cols = ['SpeedSlow', 'SpeedFastLapse', 'SpeedFastVariable']
    speed_df = pd.DataFrame(scores, columns = speed_pc_cols)

    plot_correlation_matrix(resid[speed], speed, figsize=[3.3*1.75, 3.3*1.5])
    plot_grouped_bars(evecs.T, xticklabels=speed, ylabel='Loadings', rotation = 30, figsize=[3.3*3,3])
    plot_scree_cumsum(evals, pc_names = speed_pc_cols)

    PCDF = pd.concat([affect_df, atimes_df, cog_df, speed_df], axis=1)
    PCDF = age.join(PCDF).join(resid[['aes_rt_lapse','phq_rt_lapse','lat_med']]).reset_index(drop=True)

    # Run linear model predicting endogenous from PCDF
    models    = run_lms(endog, PCDF)
    regcnames = PCDF.columns.to_list()


    pvals = False
    plot_lm_results(models, cols= ['rew_tot', 'lat_med'], cnames = regcnames, labels = ['Total Reward', 'Median Latency'], pvals=pvals)
    plot_lm_results(models, cols= ['b_h-'   , 'b_-h'   ], cnames = regcnames, labels = ['Initial Reward Rate', 'Reward Decay Rate'], pvals=pvals)
    plot_lm_results(models, cols= ['baseline_composite', 'hl_composite'], cnames = regcnames, labels = ['Overstaying', 'High-Slow Preference'], pvals=pvals)
    plot_lm_results(models, cols= ['ll_dur_dev', 'hh_dur_dev'], cnames = regcnames, labels = ['Low-Slow Deviation', 'High-Fast Deviation'], pvals=pvals)
    plot_lm_results(models, cols= ['exit_delta'], cnames = regcnames, labels = ['Exit Delta'], pvals=pvals)

    plot_correlation_matrix(PCDF, PCDF.columns.tolist(), figsize=[3.3*3, 3.3*3])

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



def single_vars(grp, exog_cols):
    pc1_scores_young = grp['dur_avg_pc1_score'].loc[grp['age_cat']==0]
    pc1_scores_old   = grp['dur_avg_pc1_score'].loc[grp['age_cat']==1]

    F, p = sp.stats.f_oneway(pc1_scores_young, pc1_scores_old)
    d = cohen_d(pc1_scores_young, pc1_scores_old)

    pc2_scores_young = grp['dur_avg_pc2_score'].loc[grp['age_cat']==0]
    pc2_scores_old   = grp['dur_avg_pc2_score'].loc[grp['age_cat']==1]

    F, p = sp.stats.f_oneway(pc2_scores_young, pc2_scores_old)
    d = cohen_d(pc2_scores_young, pc2_scores_old)

    plt.figure(figsize=[3.3,3])
    support = np.arange(-1, 1.4, 0.01)
    kde1 = sp.stats.gaussian_kde(pc1_scores_young)
    kde2 = sp.stats.gaussian_kde(pc1_scores_old)
    plot(support, kde1(support), label='Young')
    plot(support, kde2(support), label='Old')
    plt.xlabel('PC1 Scores')
    plt.ylabel('Kernel Density Estimate')
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['tol_rtc_med']/1000, grp['dur_avg_pc1_score'], c=grp['rew_tot'], cmap='viridis', s=5)
    plt.xlabel('ToL Correct RT Median [s]')
    plt.ylabel('PC1 Scores')
    plt.tight_layout()

    names = {'pc1':'PC1', 'pc2':'PC2'}
    for pc in ['pc1', 'pc2']:
        for exog in exog_cols:
            drop = np.where(grp[exog].isna())[0]
            corr = sp.stats.pearsonr(grp[exog].drop(drop), grp['dur_avg_'+pc+'_score'].drop(drop))
            print(f'Correlation between {exog} and {pc} score: {corr.statistic:0.2f}')
            print(f'p-value: {corr.pvalue:0.2e}')

            plt.figure(figsize=[3.3,3])
            plt.scatter(grp[exog].drop(drop), grp['dur_avg_'+pc+'_score'].drop(drop), c=grp['rew_tot'], cmap='viridis', s=5)
            plt.ylabel(f'{names[pc]} Scores')



def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.var(x) + (ny-1)*np.var(y)) / dof)

def plot_scree_cumsum(evals, figsize=[3.3,3], pc_names = None):
    plt.figure(figsize=figsize); 
    plt.plot(np.cumsum(evals)/sum(evals), '-o', zorder=10)
    if pc_names is not None:
        plt.xticks(np.arange(len(pc_names)), pc_names, rotation=30, ha='right')
    plt.grid()
    plt.tight_layout()


def plot_age_lm_results(models, figsize=[3.3*3,3], pvals=False):
    cols = models.columns.to_list()
    xs   = np.arange(len(cols))

    # Plot COEFFs for AGE vs EXOG
    plt.figure(figsize=figsize)
    plt.errorbar(xs, models.loc['standard_coef'], 2*models.loc['standard_err'], marker='o', linestyle='None', zorder=10)
    plt.xticks(xs, cols, rotation=30, ha='right')
    plt.ylim([-0.5,0.5])
    plt.grid()
    plt.legend(['Beta +- 2 SE'])
    plt.ylabel('Standard Coeff.')
    plt.tight_layout()

    # Plot P-VALUES for AGE vs EXOG
    if pvals:
        plt.figure(figsize=figsize)
        plt.scatter(xs, np.log(models.loc['p-value']), marker='o', linestyle='None', zorder=10)
        plt.xticks(xs, cols, rotation=30, ha='right')
        plt.grid()
        plt.ylabel('Log p-values')
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


def plot_lm_results(model, cols, cnames, labels, pvals = False, figsize=[3.3*3,3]):
    xs = np.arange(len(cnames))
    if len(cols) == 1:
        offsets = [0]
    elif len(cols) == 2:
        offsets = [-0.1, 0.1]

    # Regression coefficients
    plt.figure(figsize=figsize)
    for i, col in enumerate(cols):
        plt.errorbar(xs + offsets[i], model[col].loc['standard_coef'], 2*model[col].loc['standard_err'], marker='o', linestyle='None', zorder=10)
    plt.xticks(xs, cnames, rotation=30, ha='right')
    plt.ylim([-0.5,0.5])
    plt.grid()
    plt.legend(labels)
    plt.ylabel('Standard Coeff.')
    plt.tight_layout()

    if pvals:
        # P-values
        plt.figure(figsize=figsize)
        for i, col in enumerate(cols):
            plt.scatter(xs - offsets[i], np.log(model[col].loc['p-value']), marker='o', linestyle='None', zorder=10)
        plt.xticks(xs, cnames, rotation=30, ha='right')
        plt.grid()
        plt.legend(labels)
        plt.ylabel('Log p-values')
        plt.tight_layout()

def plot_regressor_pca(exog):
    # PCA of regressors
    evals, evecs, scores = PCA(exog)
    plot_grouped_bars(evecs[:,0:5].T, xticklabels=exog.columns.to_list(), ylabel='Loadings', rotation = 30, figsize=[13.2,3])


def plot_policy_prediction_lms_disaggregated(grp):

    # Regressors for primary outcome variables, excluding age
    exog_cols = [
        'tol_acc',
        'aes_rt_med'  , 'phq_rt_med'  , 'tol_rtc_med'  , 'tol_rti_med'  , 'rtt_rt_med',
        'aes_rt_mad'  , 'phq_rt_mad'  , 'tol_rtc_mad'  , 'tol_rti_mad'  , 'rtt_rt_mad',
        'aes_rt_lapse', 'phq_rt_lapse', 'tol_rtc_lapse', 'tol_rti_lapse', 'rtt_rt_lapse',
        'lat_med']
    
    aes_qs = ['AES_'+str(i) for i in range(17)]
    phq_qs = ['PHQ_'+str(i) for i in range(9)]

    exog_cols = exog_cols + aes_qs + phq_qs

    # We use these after we note MADs and lapse rates are not predictive
    # Regressors for primary outcome variables, excluding age
    reduced_exog = exog_cols

    # Things we try to predict
    #endog_cols = ['dur_scale_resid', 'rew_tot','b_const', 'b_h-', 'b_-h', 'b_hl']
    #endog_cols = ['log_exit_means', 'be_const', 'be_h-','be_-h', 'be_hl']
    endog_cols = [
        'rew_tot', 'lat_med',
         'b_h-', 'b_-h',
        'baseline_composite', 'hl_composite',
        'll_dur_dev', 'hh_dur_dev', 'lh_dur_dev',
        'exit_delta'
        ]

    # Linear regressions for exogenous variables by age
    exog_resid = pd.DataFrame()
    age_models = pd.DataFrame(np.zeros([4,len(exog_cols)]), ['standard_coef', 'standard_err', 'p-value', 'r-squared'], columns = exog_cols)
    for col in exog_cols:

        # Enforce float dtype
        grp[col] = grp[col].astype(float)
        
        # Get LMs for predicting each exogenous variable by age
        res = run_lm(grp[col], grp['age'], zscore = True, robust = False, add_const = True, disp=False)

        # Collect results (we'll plot them below)
        age_models.loc['standard_coef',col] = res.params[1]
        age_models.loc['standard_err', col] = res.bse[1]
        age_models.loc['p-value'      ,col] = res.pvalues[1]
        age_models.loc['r-squared'    ,col] = res.rsquared

        # Residualize
        exog_resid[col] = get_resid(grp[col], grp['age'], disp = False)

    # Linear regressions for outcomes by residuals
    model_resid = {}
    reduced_cnames = ['age'] + reduced_exog
    for col in endog_cols:
        model_resid[col] = pd.DataFrame(np.zeros([3,len(reduced_cnames)]), ['standard_coef', 'standard_err', 'p-value',], columns = reduced_cnames)

        # Enforce float dtype
        grp[col] = grp[col].astype(float)
        
        # Get LMs for predicting each exogenous variable by age
        resid = grp[['age']].join(exog_resid[reduced_exog])

        # If endogeneous variable is lat_med, remove from predictors
        if col == 'lat_med':
            resid = resid.drop('lat_med', axis=1)

        # # Z-score residuals
        # for i in resid.columns:
        #     resid[i] = robust_zscore(resid[i])

        # # Transform to PCA coordinates
        # evals, evecs = np.linalg.eig(np.cov(resid.T))
        # resid = pd.DataFrame(np.matmul(resid.values, evecs), columns=resid.columns.tolist())

        res = run_lm(grp[col], resid, zscore = True, robust = False, add_const = True, disp=True)

        # Collect results (we'll plot them below)
        model_resid[col].loc['standard_coef'] = res.params[1:]
        model_resid[col].loc['standard_err' ] = res.bse[1:]
        model_resid[col].loc['p-value'      ] = res.pvalues[1:]

    # Linear regressions for outcomes by non-residualized variables
    model = {}
    reduced_cnames = ['age'] + reduced_exog
    for col in endog_cols:
        model[col] = pd.DataFrame(np.zeros([3,len(reduced_cnames)]), ['standard_coef', 'standard_err', 'p-value',], columns = reduced_cnames)

        # Get LMs for predicting each exogenous variable by age
        X = grp[['age']].join(grp[reduced_exog])
        res = run_lm(grp[col], X, zscore = True, robust = False, add_const = True, disp=False)

        # Collect results (we'll plot them below)
        model[col].loc['standard_coef'] = res.params[1:]
        model[col].loc['standard_err' ] = res.bse[1:]
        model[col].loc['p-value'      ] = res.pvalues[1:]



    # Reward and latency
    plt.figure(figsize=[3.3*3,3])
    xs = np.arange(len(reduced_cnames))    
    col = 'rew_tot';  plt.errorbar(xs - 0.1, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', linestyle='None', zorder=10, label='Total Reward')
    col = 'lat_med';  plt.errorbar(xs + 0.1, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', c = 'r', linestyle='None', zorder=10, label='Median Latency')
    plt.xticks(xs, reduced_cnames, rotation=30, ha='right')
    plt.grid()
    plt.legend()
    plt.ylabel('Standard Coeff.')
    plt.tight_layout()

    # Condition responses
    plt.figure(figsize=[3.3*3,3])
    xs = np.arange(len(reduced_cnames))    
    col = 'b_h-';  plt.errorbar(xs - 0.1, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', linestyle='None', zorder=10, label='Initial Reward Rate')
    col = 'b_-h';  plt.errorbar(xs + 0.1, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', c = 'r', linestyle='None', zorder=10, label='Reward Decay Rate')
    plt.xticks(xs, reduced_cnames, rotation=30, ha='right')
    plt.grid()
    plt.legend()
    plt.ylabel('Standard Coeff.')
    plt.tight_layout()


    # Over-staying, High-Slow preference
    plt.figure(figsize=[3.3*3,3])
    xs = np.arange(len(reduced_cnames))    
    col = 'baseline_composite';  plt.errorbar(xs - 0.1, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', linestyle='None', zorder=10, label='Over-Staying')
    col = 'hl_composite';        plt.errorbar(xs + 0.1, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', c = 'r', linestyle='None', zorder=10, label='High-Slow Preference')
    plt.xticks(xs, reduced_cnames, rotation=30, ha='right')
    plt.grid()
    plt.legend()
    plt.ylabel('Standard Coeff.')
    plt.tight_layout()


    # Other deviations
    plt.figure(figsize=[3.3*3,3])
    xs = np.arange(len(reduced_cnames))    
    col = 'll_dur_dev';  plt.errorbar(xs - 0.2, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', linestyle='None', zorder=10, label='Low-Slow Deviation')
    col = 'lh_dur_dev';  plt.errorbar(xs + 0.0, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', c = 'r', linestyle='None', zorder=10, label='Low-Fast Deviation')
    #col = 'hh_dur_dev';  plt.errorbar(xs + 0.2, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', c = 'm', linestyle='None', zorder=10, label='High-Fast Deviation')
    plt.xticks(xs, reduced_cnames, rotation=30, ha='right')
    plt.grid()
    plt.legend()
    plt.ylabel('Standard Coeff.')
    plt.tight_layout()


    # Exit delta
    plt.figure(figsize=[3.3*3,3])
    xs = np.arange(len(reduced_cnames))    
    col = 'exit_delta';  plt.errorbar(xs - 0.2, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', linestyle='None', zorder=10, label='Exit Inconsistency')
    #col = 'hh_dur_dev';  plt.errorbar(xs + 0.2, model_resid[col].loc['standard_coef'], 2*model_resid[col].loc['standard_err'], marker='o', c = 'm', linestyle='None', zorder=10, label='High-Fast Deviation')
    plt.xticks(xs, reduced_cnames, rotation=30, ha='right')
    plt.grid()
    plt.legend()
    plt.ylabel('Standard Coeff.')
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
        plt.savefig('./data/figs/hist_'+col+'.png', dpi=120)
        plt.close()



# Plot policy heat map
def plot_policy_sweep_heatmap(grp, figsize = [3.3,3], show_title = False):
    
    # Get rewards for a bunch of policies 
    deltas = np.arange(-2500, 8500, 1e2)    # Baseline changes
    scales = np.arange(0, 2.4, 0.1)         # Multiplicative factor on modulation

    rewards = run_policy_sweep(deltas, scales)
    
    # Rescale units on deltas
    deltas = deltas*1e-3
    
    figure(figsize=[3.3,3])
    #xticks = np.arange(0, 20, 2)
    #yticks = np.arange(0, 40, 4)

    #imshow(rewards, origin='lower', aspect='auto', interpolation='none', cmap = 'Blues')

    #ax = gca();
    #ax.set_xticks(xticks);
    #ax.set_yticks(yticks);
    #ax.set_xticklabels(np.round(scales[xticks], decimals = 1 ));
    #ax.set_yticklabels(deltas[yticks]);

    # Get meshgrid. Function evaluations need to be at centers.
    xinc = scales[1]-scales[0]
    yinc = deltas[1]-deltas[0]
    
    xmin, xmax = min(scales)-xinc/2, max(scales)+xinc/2
    ymin, ymax = min(deltas)-yinc/2, max(deltas)+yinc/2
    
    xx, yy = np.mgrid[xmin:xmax:xinc, ymin:ymax:yinc]

    # Contour plot
    cset = plt.contour(xx, yy, rewards.T, levels = range(19,27))
    plt.clabel(cset, inline=1, fontsize=10)

    # Color mesh plot
    #plt.pcolormesh(xx, yy, sweep_rewards.T)
    
    xlim([0, 2.2])
    ylim([ymin, ymax])
    
    xlabel('Scale')
    ylabel('Offset [s]')
    if show_title:
        title('Policy Differences')

        # Get optimal policy
    policy_opt = get_policy_opt()
    
    # Get subject coordinates
    subj_base, subj_delta, subj_mod = policy_space(policy_opt, grp)

    # Plot them
    plt.scatter(subj_mod, subj_delta*1e-3, zorder=10, c = grp.rew_tot, s=7.5)
    tight_layout()

    corr = sp.stats.pearsonr(subj_mod, subj_delta)
    print(f'Correlation between scaling and offset: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')

    corr = sp.stats.pearsonr(subj_delta, grp['rew_tot'])
    print(f'Correlation between offset and reward: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')

    corr = sp.stats.pearsonr(subj_mod, grp['rew_tot'])
    print(f'Correlation between scaling and reward: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')

    resid = get_resid(subj_mod, subj_delta)
    corr = sp.stats.pearsonr(resid, grp['rew_tot'])
    print(f'Correlation between scaling residual and reward: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')


def plot_avg_dur_vs_reward(grp):
    cols = ['hl_dur_avg', 'll_dur_avg','hh_dur_avg', 'lh_dur_avg']
    rewards = grp['rew_tot']
    avg_dur = np.mean(grp[cols], axis=1)
    plt.figure(figsize=[3.3,3])
    plt.scatter(avg_dur/1000, rewards, c = rewards, cmap = 'viridis', s=5)
    plt.xlabel('Average Stay Duration [s]')
    plt.ylabel('Total Reward')
    cbar = plt.colorbar()
    cbar.set_label('Total Reward', rotation=270, va='bottom')
    plt.tight_layout()


def plot_avg_lat_vs_reward(grp):
    cols = ['hl_lat_avg', 'll_lat_avg', 'hh_lat_avg', 'lh_lat_avg']
    rewards = grp['rew_tot']
    avg_dur = np.mean(grp[cols], axis=1)
    plt.figure(figsize=[3.3,3])
    plt.scatter(avg_dur/1000, rewards, c = rewards, cmap = 'viridis', s=5)
    plt.xlabel('Average Stay Duration [s]')
    plt.ylabel('Total Reward')
    cbar = plt.colorbar()
    cbar.set_label('Total Reward', rotation=270, va='bottom')
    plt.tight_layout()


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

