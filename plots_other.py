from configs import *
from analy   import *
from utils   import *

from plot_lms import plot_lm_results

from reads import drop_subjects
import statsmodels.formula.api as smf

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
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)



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


def plot_group_ttest_table(grp, vars, excl=[], split='age', thresh=50, fname=None):

    # Initialize drop list with any exclusion rows passed in
    droplist = np.array(excl)

    # Add any nan rows to droplist
    no_exog = np.where(grp[vars].isna().any(axis=1))[0]
    no_age  = np.where(grp[['age']].isna().any(axis=1))[0]
    droplist = np.unique(np.concatenate([no_exog, no_age, droplist]))

    # Drop rows, report
    n_initial = len(grp)
    _, grp = drop_subjects([], grp, droplist, reset_index=True)
    n_final = len(grp)
    print('Dropped', n_initial-n_final, 'subjects for failing exogenous quality controls.')
    print('Proportion subjects remaining:', n_final/n_initial)

    rows = []
    for var in vars:

        # Need to change the units on several variables, should have done it elsewhere but fixing is involved
        if var in ['hl_exit_avg', 'll_exit_avg', 'hh_exit_avg', 'lh_exit_avg']:
            grp[var] = grp[var] * 1e2 # Convert to 1/s

        if var in ['rtt_rt_med', 'tol_rtc_med', 'dur_avg_offset', 'dur_h0_offset', 'dur_h1_offset']:
            grp[var] = grp[var] / 1e3 # Convert to seconds

        # -----------------------------
        # Split data for Younger vs. Older
        # -----------------------------
        if split != var:
            g3 = grp[[split, var]].copy()
        else:
            g3 = grp[[split]].copy()

        data_g1 = g3.loc[g3[split] <= thresh, var]
        data_g2 = g3.loc[g3[split]  > thresh, var]
        
        mean_g1 = data_g1.mean()
        sd_g1 = data_g1.std(ddof=1)
        sem_g1 = sd_g1 / np.sqrt(len(data_g1)) if len(data_g1) > 1 else np.nan
        
        mean_g2 = data_g2.mean()
        sd_g2 = data_g2.std(ddof=1)
        sem_g2 = sd_g2 / np.sqrt(len(data_g2)) if len(data_g2) > 1 else np.nan
        
        # ---------------------------------------------
        # Perform an independent t-test, unequal variance
        # ---------------------------------------------
        t_stat, p_val = sp.stats.ttest_ind(data_g1, data_g2, equal_var=False)

        # ---------------------------------------------
        # Format strings as "mean ± SEM (SD)" in sci. notation
        # ---------------------------------------------
        g1_str = f"{mean_g1:3.4f} ± {sem_g1:3.4f} ({sd_g1:3.4f})"
        g2_str = f"{mean_g2:3.4f} ± {sem_g2:3.4f} ({sd_g2:3.4f})"
        t_str  = f"{t_stat:3.4f}"
        p_str  = f"{p_val:.1e}"
        df_str = f"{len(data_g1) + len(data_g2) - 2}"
        ny_str = f"{len(data_g1)}"
        no_str = f"{len(data_g2)}"

        # -----------------------------------------------------------
        # If p < 0.05, make the *mean* portion of each group bold.
        # We do this by splitting off the text before " ± ".
        # The mean portion is then wrapped in mathtext $\mathbf{...}$.
        # -----------------------------------------------------------
        if p_val < 0.05:
            # Younger
            try:
                mean_part_g1, rest_g1 = g1_str.split(" ± ", 1)
                # Use a raw string f"r" and \mathbf{} for bold in mathtext
                g1_str  = rf"{mean_part_g1} ± {rest_g1}"
            except ValueError:
                pass  # fallback if something unexpected in the string

            # Older
            try:
                mean_part_g2, rest_g2 = g2_str.split(" ± ", 1)
                g2_str  = rf"{mean_part_g2} ± {rest_g2}"
            except ValueError:
                pass
        
        rows.append([var, g1_str, g2_str, t_str, p_str, df_str, ny_str, no_str])

    # ------------------------------------------------
    # Build table DataFrame for display
    # ------------------------------------------------
    table_df = pd.DataFrame(
        rows,
        columns=["Variable", "Younger, Mean +- SEM (SD)", "Older, Mean +- SEM (SD)", "t-stat", "p-value", "df", "n Young", "n Old"]
    )

    # ----------------------------------------------
    # Render the DataFrame as a matplotlib table
    # ----------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 2 + 0.2*len(table_df)))
    ax.axis("off")  # turn off default axis spines/ticks

    tab = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center"
    )

    # Adjust the table layout
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.auto_set_column_width(col=list(range(len(table_df.columns))))
    plt.tight_layout()
    plt.show()

    # Print the table to output
    print(table_df)

    # Save the table to a figure file if requested
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

    return droplist



def plot_lme_of_stay_by_age_and_decay_rate(subj, grp, droplist, type='dur'):
    n_initial = len(grp)
    
    subj, grp = drop_subjects(subj, grp, droplist, reset_index=True)

    n_final = len(grp)
    print('Dropped', n_initial-n_final, 'subjects for failing exogenous quality controls.')
    print('Proportion subjects remaining:', n_final/n_initial)


    policy_opt = get_policy_opt()

    # Number of subjects
    ns = len(subj)
    
    # Beta lists (to frame later)
    cols = ['b_const', 'b_-h', 'b_age', 'p_const', 'p_-h', 'p_age', 'r2']
    cols = [type + '_' + col for col in cols]
    results  = pd.DataFrame(np.zeros([1, len(cols)]), columns = cols)

    # Loop through subjects getting their betas
    for s in range(0, ns):
        
        # Get subject duration data
        X = subj[s][['hl','ll','hh','lh']]
    
        optimal_stay = X @ policy_opt

        if type == 'dur':
            dev = subj[s]['space_down_time'] - optimal_stay
        else:
            dev = subj[s]['exit']*100
        # Boolean predictor columns for init and decay
        age_cat = 1 if grp['age'][s] > 50 else 0
        sid = np.repeat(s, len(X))
        X2 = pd.DataFrame( [X['hh'] + X['lh'], np.repeat(age_cat, len(X)), sid, dev], index=['_h', 'age', 'sid', 'dev']).T

        # Add intercept to model
        X2 = sm.add_constant(X2, has_constant='add')

        if s == 0:
            X3 = X2.copy()
        else:
            X3 = pd.concat([X3, X2], axis=0)

    X3.reset_index(drop=True, inplace=True)
    X3['sid'] = X3['sid'].astype('int')

    md = smf.mixedlm("dev ~ _h + age + _h*age", X3, groups=X3["sid"])
    res = md.fit()
    print(res.summary())

    plt.figure()
    plt.errorbar(res.params.index.values, res.params.values, 2*res.bse, marker='o', linestyle='')
    plt.grid()
    plt.xlabel('Parameter')
    plt.ylabel('Estimate')
    if type == 'dur':
        plt.title('Mixed model of stay durations')
    else:
        plt.title('Mixed model of exit thresholds')
    plt.xticks(rotation=45)
    plt.legend(['Estimate +- 2*SE'])
    plt.show()
    plt.tight_layout()


def plot_offset_and_scale_by_time_and_age(grp, var='dur', t0='h0', t1='h1', item='offset', fname=None):
    # Duration offsets
    scale = 1e-3 if (item == 'offset') & (var == 'dur') else 1

    rows = grp['age_cat']==0
    data_t0 = grp[var+'_'+t0+'_'+item][rows]*scale
    data_t1 = grp[var+'_'+t1+'_'+item][rows]*scale
    beg_young = np.mean(data_t0)
    end_young = np.mean(data_t1)
    beg_young_se = np.std(data_t0)/np.sqrt(len(data_t0))
    end_young_se = np.std(data_t1)/np.sqrt(len(data_t1))

    rows = grp['age_cat']==1
    data_t0 = grp[var+'_'+t0+'_'+item][rows]*scale
    data_t1 = grp[var+'_'+t1+'_'+item][rows]*scale
    beg_old = np.mean(data_t0)
    end_old = np.mean(data_t1)
    beg_old_se = np.std(data_t0)/np.sqrt(len(data_t0))
    end_old_se = np.std(data_t1)/np.sqrt(len(data_t1))

    plt.figure(figsize=[3.3,3])
    offset = 0.025
    loc    = np.array([0,1])
    plt.errorbar(loc-offset, [beg_young, end_young], [2*beg_young_se, 2*end_young_se], marker='o', linestyle='', label='Mean +- 2SE, Young', color='m')
    plt.errorbar(loc+offset, [beg_old, end_old]    , [2*beg_old_se, 2*end_old_se]    , marker='o', linestyle='', label='Mean +- 2SE, Old'  , color='r')
    plt.xticks(loc, ['Initial', 'Final'])
    plt.xlabel('Split Half')
    if item == 'offset':
        plt.ylabel('Mean Duration Offset [s]')
    elif item == 'scale':
        plt.ylabel('Mean Duration Scale')
    plt.legend()

    # Rotate xtick labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)


def plot_pooled_stay_durations_by_age(grp, fname=None):
    avg = grp[['-h_dur_avg', '-l_dur_avg']]/1000
    avg_young = np.mean(avg[grp['age_cat'] == 0],axis=0)
    avg_old   = np.mean(avg[grp['age_cat'] == 1],axis=0)
    sem_young = np.std(avg[grp['age_cat'] == 0])/np.sqrt(np.sum(grp['age_cat'] == 0))
    sem_old   = np.std(avg[grp['age_cat'] == 1])/np.sqrt(np.sum(grp['age_cat'] == 1))

    plt.figure(figsize=[3.3,3])
    offset=0.025
    plt.errorbar([0-offset, 1-offset], avg_young, 2*sem_young, marker='o', linestyle='', label='Mean +- 2*SE, Young', color='m')
    plt.errorbar([0+offset, 1+offset], avg_old  , 2*sem_old  , marker='o', linestyle='', label='Mean +- 2*SE, Old'  , color='r')
    plt.xticks([0,1], ['Fast', 'Slow'])
    plt.xlabel('Pooled Decay Rate Conditions')
    plt.ylabel('Stay Duration [s]')
    plt.legend()
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)

def plot_pc_scores_by_age(grp, fname=None):
    avg = grp[['dur_avg_pc1_score', 'dur_avg_pc2_score']]
    avg_young = np.mean(avg[grp['age_cat'] == 0], axis=0)
    avg_old   = np.mean(avg[grp['age_cat'] == 1], axis=0)
    sem_young = np.std(avg[grp['age_cat'] == 0])/np.sqrt(np.sum(grp['age_cat'] == 0))
    sem_old   = np.std(avg[grp['age_cat'] == 1])/np.sqrt(np.sum(grp['age_cat'] == 1))

    plt.figure(figsize=[3.3,3])
    offset = 0.025
    plt.errorbar([0-offset, 1-offset], avg_young, 2*sem_young, marker='o', linestyle='', label='Mean +- 2*SE, Young', color='m')
    plt.errorbar([0+offset, 1+offset], avg_old  , 2*sem_old  , marker='o', linestyle='', label='Mean +- 2*SE, Old'  , color='r')
    plt.xticks([0,1], ['PC1', 'PC2'])
    plt.xlabel('Principal Components')
    plt.ylabel('Scores')
    plt.legend()
    plt.tight_layout()
    if fname is not None: plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)


def plot_policy_prediction_lms_without_residualizing_age(grp, fname=None):

    exog_cols = ['aes_score', 'phq_score', 'tol_acc', 'tol_rtc_med', 'rtt_rt_med']

    # Things we try to predict
    endog_cols  = ['dur_avg_pc1_score', 'dur_avg_pc2_score']
    endog_names = ['Duration PC1', 'Duration PC2']

    # Predictor / design matrix (without constant)
    exog = grp[exog_cols].loc[grp['exg_qc']].copy()
    age  = grp[['age']].loc[grp['exg_qc']].copy()

    # Find any locations where there are NaNs
    no_exog = np.where(exog.isna().any(axis=1))[0]
    no_age  = np.where(age.isna().any(axis=1))[0]
    keep    = np.setdiff1d(np.arange(len(exog)), np.union1d(no_exog, no_age))

    # Subset data
    exog = exog.iloc[keep].reset_index(drop=True)
    age  = age.iloc[keep].reset_index(drop=True)

    # Run linear model predicting endogenous from residuals and age
    endog     = grp[endog_cols].copy().iloc[keep].reset_index(drop=True) 
    models    = run_lms(endog, age.join(exog))
    regcnames = ['age'] + exog_cols

    i = 0
    sfx = ['b', 'c']
    for endog_col, name in zip(endog_cols, endog_names):
        plot_lm_results(models, cols=[endog_col], cnames = regcnames, labels = [name], figsize=[3.3*1,3], legend=False)
        if fname is not None:
            fname = fname+sfx[i]
            plt.savefig(save_path+fname+'.'+fmt, dpi=dpi, format=fmt)
        i += 1



def plot_lme_of_offset_by_split_half_and_age(grp):
    n_initial = len(grp)
    
    # Drop subjects with NA age
    droplist = np.where(grp['age'].isna())[0]

    _, grp = drop_subjects([], grp, droplist, reset_index=True)

    n_final = len(grp)
    print('Dropped', n_initial-n_final, 'subjects for having no age value.')
    print('Proportion subjects remaining:', n_final/n_initial)

    n_subj = len(grp)

    age_var = 'age'
    dur_var = 'scale'

    h0_augmented = np.concatenate([grp[[age_var,'sid']], np.repeat([[0]], n_subj,axis=0)],axis=1)
    h1_augmented = np.concatenate([grp[[age_var,'sid']], np.repeat([[1]], n_subj,axis=0)],axis=1)

    h0_augmented = pd.DataFrame(h0_augmented, columns=[age_var,'sid','h'])
    h1_augmented = pd.DataFrame(h1_augmented, columns=[age_var,'sid','h'])

    X = sm.add_constant(pd.concat([h0_augmented, h1_augmented],axis=0))
    Y = pd.concat([grp['dur_h0_' + dur_var], grp['dur_h1_' + dur_var]], axis=0)

    # Reindex    
    Y.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)

    # Set column name for Y
    Y.name = dur_var

    # Create one dataframe for LME model
    df = pd.concat([X, Y], axis=1)

    # Drop NaN rows
    df = df.dropna(axis=0, how='any')
    df = df.reset_index(drop=True)
    df['sid']   = df['sid'  ].astype('int')
    df['h']     = df['h'    ].astype('int')
    df[age_var] = df[age_var].astype('int')

    md = smf.mixedlm(dur_var+" ~ h + "+age_var+" + h*"+age_var, df, groups=df["sid"])
    res = md.fit()
    print(res.summary())

    plt.figure()
    plt.errorbar(res.params.index.values, res.params.values, 2*res.bse, marker='o', linestyle='')
    plt.grid()
    plt.xlabel('Parameter')
    plt.ylabel('Estimate')
    plt.title('Mixed model of stay duration ' + dur_var)

    plt.xticks(rotation=45)
    plt.legend(['Estimate +- 2*SE'])
    plt.show()
    plt.tight_layout()