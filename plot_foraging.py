from configs    import *
from analy      import *
from utils      import *
from plot_utils import *

def plot_reward_rate_conds(fname=None):
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
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)


def plot_reward_cumulative_conds(fname=None):
    t = np.arange(0,20000)
    conds = ((0.02, 0.0002), (0.01, 0.0002), (0.02, 0.0004), (0.01, 0.0004))

    dmdt, rews = [],[]
    for cond in conds:
        dmdt.append(cond[0]* np.exp(-cond[1]*t))
        rews.append(1/100 * cond[0] / cond[1] * (1 - np.exp(-cond[1] * t)))

    # Accumulated reward curves
    plt.figure(figsize=[3.3,3])
    for i in range(4):
        plt.plot(t/1000, rews[i])

    plt.xlabel('Time [s]')
    plt.ylabel('Total Reward')
    plt.legend(['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast'])
    plt.tight_layout()

    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

def plot_durations(grp, by_age = False, show_title = False, scatter=True, fname=None):
    
    # Get optimal and subject policies
    policy_opt = get_policy_opt()*1e-3
    avg = get_policies(grp)*1e-3

    # Number of subjects
    ns = np.shape(grp)[0]
    
    # Color subjects by optimality
    cmap = 'viridis'

    # Figure
    plt.figure(figsize = [3.3,3])

    # Optimal policy * group averages
    plt.plot([0,1,2,3], policy_opt, '--ok')

    if by_age:
        avg_young = np.mean(avg[grp['age_cat'] == 0],axis=0)
        avg_old   = np.mean(avg[grp['age_cat'] == 1],axis=0)
        sem_young = np.std(avg[grp['age_cat'] == 0])/np.sqrt(np.sum(grp['age_cat'] == 0))
        sem_old   = np.std(avg[grp['age_cat'] == 1])/np.sqrt(np.sum(grp['age_cat'] == 1))

        plt.errorbar([0,1,2,3], avg_young , 2*sem_young, linestyle='--', marker='o', color='m')
        plt.errorbar([0,1,2,3], avg_old   , 2*sem_old  , linestyle='--', marker='o', color='r')
    else:
        plt.plot([0,1,2,3], np.mean(avg,0) , '--or')

    # Subject data
    if scatter:
        plt.scatter(np.random.normal(0,0.1,ns), avg.iloc[:,0], c = grp.rew_tot, cmap = cmap, s = 5)
        plt.scatter(np.random.normal(1,0.1,ns), avg.iloc[:,1], c = grp.rew_tot, cmap = cmap, s = 5)
        plt.scatter(np.random.normal(2,0.1,ns), avg.iloc[:,2], c = grp.rew_tot, cmap = cmap, s = 5)
        plt.scatter(np.random.normal(3,0.1,ns), avg.iloc[:,3], c = grp.rew_tot, cmap = cmap, s = 5)

    if by_age:
        plt.legend(['Optimal', 'Mean +- 2SE, Young', 'Mean +- 2SE, Old'])
    else:
        plt.legend(['Optimal', 'Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = plt.gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 30)
    
    plt.ylabel('Stay Durations [s]')

    if scatter:
        plt.ylim([0,20])

    if show_title:
        plt.title('Subject Stay-Durations')

    if scatter:
        cbar = plt.colorbar()
        cbar.set_label('Total Reward', rotation=270, va='bottom')
    plt.tight_layout()

    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

    # Print statistics
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


def plot_avg_dur_vs_reward(grp, fname=None):
    cols = ['hl_dur_avg', 'll_dur_avg','hh_dur_avg', 'lh_dur_avg']
    rewards = grp['rew_tot']
    avg_dur = np.mean(grp[cols], axis=1)
    plt.figure(figsize=[3.3,3])
    plt.scatter(avg_dur/1000, rewards, c = rewards, cmap = 'viridis', s=5)
    plt.xlabel('Average Stay Duration [s]')
    plt.ylabel('Total Reward')
    #cbar = plt.colorbar()
    #cbar.set_label('Total Reward', rotation=270, va='bottom')
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

def plot_avg_lat_vs_reward(grp, fname=None):
    cols = ['hl_lat_avg', 'll_lat_avg', 'hh_lat_avg', 'lh_lat_avg']
    rewards = grp['rew_tot']
    avg_lat = grp['lat_avg']/1000
    plt.figure(figsize=[3.3,3])
    plt.scatter(avg_lat, rewards, c = rewards, cmap = 'viridis', s=5)
    plt.xlabel('Average Response Latency [s]')
    plt.ylabel('Total Reward')
    #cbar = plt.colorbar()
    #cbar.set_label('Total Reward', rotation=270, va='bottom')
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

    corr = sp.stats.pearsonr(avg_lat, grp['rew_tot'])
    print(f'Correlation between latency and reward: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')

    corr = sp.stats.pearsonr(avg_lat, grp['dur_avg'])
    print(f'Correlation between latency and stay duration: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')

def plot_exits(grp, by_age = False, show_title = False, fname=None):
    
    # Get optimal and subject policies
    policy_opt = get_policy_opt()
    thresh_opt = get_policy_exit_thresholds(policy_opt)*1e2

    policy_avg = get_policies(grp)
    avg = get_policy_exit_thresholds(policy_avg)

    # Number of subjects
    ns = np.shape(grp)[0]
    
    # Color subjects by optimality
    cmap = 'viridis'
    
    # Change units to [s]
    avg = avg*1e2

    # Figure
    plt.figure(figsize = [3.3,3])

    # Optimal policy * group averages
    plt.plot([0,1,2,3], thresh_opt, '--ok')

    plt.plot([0,1,2,3], np.mean(avg,0) , '--or')

    # Subject data
    plt.scatter(np.random.normal(0,0.1,ns), avg[:,0], c = grp.rew_tot, cmap = cmap, s = 5)
    plt.scatter(np.random.normal(1,0.1,ns), avg[:,1], c = grp.rew_tot, cmap = cmap, s = 5)
    plt.scatter(np.random.normal(2,0.1,ns), avg[:,2], c = grp.rew_tot, cmap = cmap, s = 5)
    plt.scatter(np.random.normal(3,0.1,ns), avg[:,3], c = grp.rew_tot, cmap = cmap, s = 5)

    plt.legend(['Optimal', 'Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = plt.gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 30)
    
    plt.ylabel('Exit Thresholds [1/s]')

    #ylim([0,20])
    if show_title:
        plt.title('Subject Exit Thresholds')
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

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


def plot_exit_devs(grp, by_age = False, show_title = False, sfx='', fname=None):
    
    devs = grp[['hl_exit_avg_dev'+sfx, 'll_exit_avg_dev'+sfx,'hh_exit_avg_dev'+sfx, 'lh_exit_avg_dev'+sfx]].values*1e2
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
    plt.ylabel('Exit Threshold Dev.s [1/s]')
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

    compare_means(devs[:,0], devs[:,1], labels=['High-Slow', 'Low-Slow'])
    compare_means(devs[:,1], devs[:,2], labels=['Low-Slow' , 'High-Fast'])
    compare_means(devs[:,2], devs[:,3], labels=['High-Fast', 'Low-Fast'])

    compare_means(devs[:,0], devs_opt[[0]], labels=['High-Slow', 'Optimal'])
    compare_means(devs[:,1], devs_opt[[1]], labels=['Low-Slow', 'Optimal'])
    compare_means(devs[:,2], devs_opt[[2]], labels=['High-Fast', 'Optimal'])
    compare_means(devs[:,3], devs_opt[[3]], labels=['Low-Fast', 'Optimal'])

    corr = sp.stats.pearsonr(grp['exit_delta_avg'], grp['exit_means_avg'])
    print(f'Correlation between exit delta and exit mean: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')

    corr = sp.stats.pearsonr(-grp['exit_delta_avg'], grp['rew_tot'])
    print(f'Correlation between exit delta and total reward: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')

    #corr = sp.stats.pearsonr(-grp['exit_avg_pc1_score'], grp['rew_tot'])
    #print(f'Correlation between pc1 score and total reward: {corr.statistic:0.2e}')
    #print(f'p-value: {corr.pvalue:0.2e}')

def plot_exit_diff_by_thresh(grp, fname=None):
    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['exit_means_avg']*1e2, grp['exit_delta_avg']*1e2, c=grp['rew_tot'], s=5)
    plt.xlabel('Exit Threshold Average [1/s]')
    plt.ylabel('Exit Threshold Max - Min [1/s]')
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)




def plot_both_ve(grp, fname=None):
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
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)


def plot_stay_pca(summary, type = 'dur', time = 'avg', fname=None):
    keys = ['_'.join([type,time,'pc'+str(i)]) for i in range(1,4)]
    pcs =   np.stack([summary[key       ] for key in keys])
    err = 2*np.stack([summary[key+'_jse'] for key in keys])

    # Plot of PCs
    plot_grouped_bars(pcs[0:2,:], xticklabels=['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast'], ylabel='Loadings', rotation = 30, figsize=[3.3,3], yerr=err)
    plt.legend(['PC1','PC2'],loc='lower left')
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

    # Plot of variance explained
    #plt.figure(figsize=[3.3,3])
    #plt.plot(np.cumsum(evals)/sum(evals), '-o', zorder=10)
    #plt.xlabel('Principal Component')
    #plt.ylabel('Cumulative Variance Explained')
    #plt.tight_layout()

    #corr = sp.stats.pearsonr(-grp['dur_pc1_score'], grp['rew_tot'])
    #print(f'Correlation between pc1 score and total reward: {corr.statistic:0.2e}')
    #print(f'p-value: {corr.pvalue:0.2e}')


# Plot policy heat map
def plot_policy_sweep_heatmap(grp, figsize = [3.3,3], show_title = False, fname=None):
    
    # Get rewards for a bunch of policies 
    deltas = np.arange(-2500, 8500, 1e2)    # Baseline changes
    scales = np.arange(0, 2.4, 0.1)         # Multiplicative factor on modulation

    rewards = run_policy_sweep(deltas, scales)
    
    # Rescale units on deltas
    deltas = deltas*1e-3
    
    plt.figure(figsize=[3.3,3])
    #xticks = np.arange(0, 20, 2)
    #yticks = np.arange(0, 40, 4)

    #imshow(rewards, origin='lower', aspect='auto', interpolation='none', cmap = 'Blues')

    #ax = plt.gca();
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
    
    plt.xlim([0, 2.2])
    plt.ylim([ymin, ymax])
    
    plt.xlabel('Scale')
    plt.ylabel('Offset [s]')
    if show_title:
        plt.title('Policy Differences')

        # Get optimal policy
    policy_opt = get_policy_opt()
    
    # Get subject coordinates
    subj_base, subj_delta, subj_mod = policy_space(policy_opt, grp)

    # Plot them
    plt.scatter(subj_mod, subj_delta*1e-3, zorder=10, c = grp.rew_tot, s=7.5)
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

    corr = sp.stats.pearsonr(subj_mod, subj_delta)
    print(f'Correlation between scaling and offset: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')

    corr = sp.stats.pearsonr(subj_delta, grp['rew_tot'])
    print(f'Correlation between offset and reward: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')

    # corr = sp.stats.pearsonr(subj_mod, grp['rew_tot'])
    # print(f'Correlation between scaling and reward: {corr.statistic:0.2f}')
    # print(f'p-value: {corr.pvalue:0.2e}')

    corr = sp.stats.pearsonr(grp['dur_avg_pc1_score'], grp['rew_tot'])
    print(f'Correlation between dur pc1 and reward: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')

    resid = get_resid(subj_mod, subj_delta, disp=False)
    corr = sp.stats.pearsonr(resid, grp['rew_tot'])
    print(f'Correlation between scaling residual and reward: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')

    corr = sp.stats.pearsonr(grp['dur_avg_pc2_score'], grp['rew_tot'])
    print(f'Correlation between dur pc2 and reward: {corr.statistic:0.2e}')
    print(f'p-value: {corr.pvalue:0.2e}')

def plot_dur_vs_exit(grp, fname=None):
    plt.figure(figsize = [3.3,3])
    plt.scatter(grp['dur_avg_pc1_score'], grp['exit_avg_pc1_score'], c=grp['rew_tot'], cmap='viridis', s=5)
    plt.xlabel('Duration PC1 Score')
    plt.ylabel('Exit PC1 Score')
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

def plot_dur_pc_scores(grp, fname=None):
    plt.figure(figsize=[3.3,3])
    plt.scatter(grp['dur_avg_pc1_score'], grp['dur_avg_pc2_score'], cmap='viridis', c = grp.rew_tot)
    plt.xlabel('Dur. PC1 Scores')
    plt.ylabel('Dur. PC2 Scores')
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)



def plot_pcs_by_time(summary, type='dur', pc='pc1', fname=None):
    # Stack PCs
    colors = {'pc1':['#336ea0', '#679fce', '#b1cee6'], 'pc2':['#d95f02', '#f4a582', '#fddbc7']}

    pcs = np.stack([summary[key] for key in [type+'_avg_'+pc, type+'_t0_'+pc, type+'_t1_'+pc]])
    err = np.stack([summary[key] for key in [type+'_avg_'+pc+'_jse', type+'_t0_'+pc+'_jse', type+'_t1_'+pc+'_jse']])

    plot_grouped_bars(pcs, xticklabels=['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast'], ylabel=pc+' Loadings', rotation = 30, figsize=[3.3,3], yerr=2*err, colors=colors[pc])
    plt.legend(['Average','Start','End'])
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

    normed_t0 = summary[type+'_t0_'+pc]/np.linalg.norm(summary[type+'_t0_'+pc])
    normed_t1 = summary[type+'_t1_'+pc]/np.linalg.norm(summary[type+'_t1_'+pc])

    angle = np.arccos(np.dot(normed_t0, normed_t1))
    print(f'Angle between {type} {pc} scores at t0 and t1: {angle*360/(2*np.pi):0.2f}')

    #corr = sp.stats.pearsonr(summary[type+'_t0_'+pc], summary[type+'_t1_'+pc])
    #print(f'Correlation between {type} {pc} score at t0 and {pc} score at t1: {corr.statistic:0.2f}')
    #print(f'p-value: {corr.pvalue:0.2e}')

def plot_exit_devs_by_time(summary, fname=None):
    # Stack PCs
    colors = ['#4d31aa', '#826ad3', '#c2b6e9']

    pcs = np.stack([summary[key] for key in ['exit_dev_means_avg', 'exit_dev_means_t0', 'exit_dev_means_t1']])*1e2
    err = np.stack([summary[key] for key in ['exit_dev_sems_avg', 'exit_dev_sems_t0', 'exit_dev_sems_t1']])*1e2

    plot_grouped_bars(pcs, xticklabels=['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast'], ylabel='Average', rotation = 30, figsize=[3.3,3], yerr=2*err, colors=colors)
    plt.legend(['Average','Start','End'])
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

    #normed_t0 = summary['exit_dev_means_t0_'+pc]/np.linalg.norm(summary['exit_dev_means_t0_'+pc])
    #normed_t1 = summary['exit_dev_means_t1_'+pc]/np.linalg.norm(summary['exit_dev_means_t1_'+pc])

    #angle = np.arccos(np.dot(normed_t0, normed_t1))
    #print(f'Angle between {type} {pc} scores at t0 and t1: {angle*360/(2*np.pi):0.2f}')

    #corr = sp.stats.pearsonr(summary['exit_dev_means_t0], summary['exit_dev_means_t1])
    #print(f'Correlation between {type} {pc} score at t0 and {pc} score at t1: {corr.statistic:0.2f}')
    #print(f'p-value: {corr.pvalue:0.2e}')

def plot_pc_scores_by_time(grp, var='dur', varname='Duration', pc='pc1', fname=None):
    pcname = pc[0:2].upper() + pc[-1]
    plt.figure(figsize=[3.3,3])
    plt.scatter(grp[var+'_h0_'+pc+'_score'], grp[var+'_h1_'+pc+'_score'], c = grp.rew_tot, cmap = 'viridis', s=5)
    plt.xlabel(f'{varname} Start {pcname} Scores')
    plt.ylabel(f'{varname} End {pcname} Scores')
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

    #good = grp[key+'_t0_pc2_score']>-1.0
    corr = sp.stats.spearmanr(grp[f'{var}_h0_'+pc+'_score'], grp[f'{var}_h1_'+pc+'_score'])
    print(f'Correlation between {var} {pc} score at h0 and {pc} score at h1: {corr.statistic:0.2f}')
    print(f'p-value: {corr.pvalue:0.2e}')


def plot_exit_dev_corrs_by_time(grp, fname=None):
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
        if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

        print(f'Deviation correlations {varname}: {corrs}')
        print(f'Deviation p-values {varname}: {pvals}')

        print(f'')
        exit_corr = sp.stats.spearmanr(grp['exit_delta_avg'], grp['exit_avg_pc1_score'])










