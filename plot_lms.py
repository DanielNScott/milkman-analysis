import numpy as np
from matplotlib import pyplot as plt
from analy      import *
from utils      import *
from plot_utils import *

save_figs = True
save_path = './figs/'
dpi = 200

def plot_policy_prediction_lms(grp):

    # Regressors for primary outcome variables, excluding age
    #exog_cols = [
    #    'aes_score'   , 'phq_score'   , 'tol_acc',
    #    'aes_rt_med'  , 'phq_rt_med'  , 'tol_rtc_med'  , 'tol_rti_med'  , 'rtt_rt_med',
    #    'aes_rt_mad'  , 'phq_rt_mad'  , 'tol_rtc_mad'  , 'tol_rti_mad'  , 'rtt_rt_mad',
    #    'aes_rt_lapse', 'phq_rt_lapse', 'tol_rtc_lapse', 'tol_rti_lapse', 'rtt_rt_lapse',
    #    'lat_med']

    exog_cols = ['aes_score', 'phq_score', 'tol_acc', 'tol_rtc_med', 'rtt_rt_med']

    # Things we try to predict
    #endog_cols  = ['rew_tot', 'dur_avg_pc1_score', 'dur_avg_pc2_score', 'll_exit_avg_dev']
    endog_cols  = ['dur_avg_pc1_score', 'dur_avg_pc2_score']
    #endog_names = ['Total Reward', 'Duration PC1', 'Duration PC2', 'Low-Slow Exit Deviation']
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

    for col in exog_cols:
        corr = sp.stats.pearsonr(np.squeeze(age), np.squeeze(exog[col]))
        print(f'Correlation between age and {col}: {corr.statistic:0.2e}')
        print(f'p-value: {corr.pvalue:0.2e}')

    # Get relationships to age and residualize
    resid, age_models = run_age_lms(exog, age)

    # Run linear model predicting endogenous from residuals and age
    endog     = grp[endog_cols].copy().iloc[keep].reset_index(drop=True) 
    models    = run_lms(endog, age.join(resid))
    regcnames = ['age'] + exog_cols

    # Age model results
    plot_age_lm_results(age_models, figsize=[3.3*1,3])
    if save_figs: plt.savefig(save_path+'fig-5a', dpi=dpi)

    #
    #plot_correlation_matrix(grp  , regcnames)
    #plot_correlation_matrix(resid, regcnames)

    #
    #plot_lm_results(models, cols= ['rew_tot', 'lat_med'], cnames = regcnames, labels = ['Total Reward', 'Median Latency'])
    #plot_lm_results(models, cols= ['b_h-', 'b_-h'], cnames = regcnames, labels = ['Initial Reward Rate', 'Reward Decay Rate'])
    #plot_lm_results(models, cols= ['baseline_composite', 'hl_composite'], cnames = regcnames, labels = ['Overstaying', 'High-Slow Preference'])
    #plot_lm_results(models, cols= ['ll_dur_dev', 'hh_dur_dev'], cnames = regcnames, labels = ['Low-Slow Deviation', 'High-Fast Deviation'])
    #plot_lm_results(models, cols= ['exit_delta'], cnames = regcnames, labels = ['Exit Delta'])

    # Predict PC1 scores from age and residuals
    # res = run_lm(endog['dur_avg_pc1_score'], age.join(resid), zscore=True)
    # preds = res.predict(sm.add_constant(sp.stats.zscore(age.join(resid))))
    # rew_tot = grp['rew_tot'].copy().iloc[keep].reset_index(drop=True) 
    # plt.figure(figsize=[3.3,3])
    # plt.scatter(sp.stats.zscore(endog['dur_avg_pc1_score']), preds, c = rew_tot, cmap = 'viridis', s=5)
    # plt.xlabel('True z(PC1 Scores)')
    # plt.ylabel('Predicted z(PC1 Scores)')

    #xlims = plt.xlim()
    #ylims = plt.ylim()
    #lower = min(xlims[0], ylims[0])
    #upper = max(xlims[1], ylims[1])
    #plt.plot([lower, upper], [lower, upper], 'k--')
    #plt.xlim([lower, upper])
    #plt.ylim([lower, upper])
    # plt.tight_layout()

    i = 0
    sfx = ['b', 'c']
    for endog_col, name in zip(endog_cols, endog_names):
        plot_lm_results(models, cols=[endog_col], cnames = regcnames, labels = [name], figsize=[3.3*1,3], legend=False)
        if save_figs: plt.savefig(save_path+'fig-5'+sfx[i], dpi=dpi)
        i += 1

    x = grp['age']
    y = grp['dur_avg_pc1_score']
    scatter_with_lm_fit(x=x, y=y, c=grp['rew_tot'], xlabel='Age', ylabel='PC1 Score')
    if save_figs: plt.savefig(save_path+'fig-5d', dpi=dpi)

    x = grp['phq_score']
    y = grp['dur_avg_pc1_score']
    scatter_with_lm_fit(x=x, y=y, c=grp['rew_tot'], xlabel='PHQ (Total Score)', ylabel='PC1 Score')
    if save_figs: plt.savefig(save_path+'fig-5e', dpi=dpi)

    x = grp['tol_rtc_med']/1000
    y = grp['dur_avg_pc1_score']
    scatter_with_lm_fit(x=x, y=y, c=grp['rew_tot'], xlabel='ToL RTc Median [s]', ylabel='PC1 Score')
    if save_figs: plt.savefig(save_path+'fig-5f', dpi=dpi)

    # x = resid['phq_score']
    # y = endog['dur_avg_pc1_score']
    # scatter_with_lm_fit(x=x, y=y, c=rew_tot, xlabel='Residualised PHQ Score', ylabel='PC1 Score')

    # x = resid['tol_rtc_med']
    # y = endog['dur_avg_pc1_score']
    # scatter_with_lm_fit(x=x, y=y, c=rew_tot, xlabel='Residualised ToL RTc Median', ylabel='PC1 Score')


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


def plot_lm_results(model, cols, cnames, labels, pvals = False, figsize=[3.3*3,3], legend=True):
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
    if legend: plt.legend(labels)
    plt.ylabel('Standard Coeff.')
    plt.tight_layout()

    if pvals:
        # P-values
        plt.figure(figsize=figsize)
        for i, col in enumerate(cols):
            plt.scatter(xs - offsets[i], np.log(model[col].loc['p-value']), marker='o', linestyle='None', zorder=10)
        plt.xticks(xs, cnames, rotation=30, ha='right')
        plt.grid()
        if legend: plt.legend(labels)
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

