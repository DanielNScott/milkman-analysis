import numpy  as np
import pandas as pd
import scipy  as sp

import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import product

from utils import *

def get_foraging_response_stats(subj, grp):
    '''Get summary statistics of subject behaviour'''
    # Loop over subjects, filling in summary statistics
    for i, subj in enumerate(subj):
        grp.at[i,'lat_avg'] = np.mean(subj.time_to_response)
        grp.at[i,'lat_med'] = np.median(subj.time_to_response)
        grp.at[i,'lat_std'] = np.std(subj.time_to_response)

        grp.at[i,'dur_avg'] = np.mean(subj.space_down_time) 
        grp.at[i,'dur_med'] = np.median(subj.space_down_time)
        grp.at[i,'dur_std'] = np.std(subj.space_down_time)

        grp.at[i,'hgt_avg'] = np.mean(subj.height)
        grp.at[i,'hgt_med'] = np.median(subj.height)
        grp.at[i,'hgt_std'] = np.std(subj.height)

        grp.at[i,'ntrials'] = subj.trial_number.iloc[-1]
        grp.at[i,'rew_tot'] = subj.total_milk.iloc[-1]
    return grp


def analyze_subjects(subj, grp):
    # Dictionary to contain summary data 1-level above grp
    summary = {}

    # Get exit thresholds and exit threshold variations
    for time_point in ['avg', 't0', 't1', 'h0', 'h1']:
        subj, grp    = get_subject_exit_thresholds(subj, grp, time = time_point, sfx='')
        grp, summary = get_subject_exit_threshold_variation(grp, summary, time=time_point)

    # Get duration, exit, height PCAs
    for time_point, var in product(['avg', 't0', 't1', 'h0', 'h1'], ['dur', 'exit', 'hgt']):
        grp, summary = get_stay_pca(grp, summary, time = time_point, var = var)

    # Get optimal policy and subject coordinates
    policy_opt = get_policy_opt()
    subj_base, subj_delta, subj_mod = policy_space(policy_opt, grp)
    grp['dur_scale'] = subj_mod
    grp['dur_scale_resid'] = get_resid(subj_mod, grp.dur_baseline, disp = False)

    # Set some duration parameters
   # grp = get_subject_dur_deviations(grp)

    # Project onto reduced models
    _ , dur_models = get_reduced_model_rve(grp, type='dur')
    grp[['hl_dur_avg_m0', 'll_dur_avg_m0','hh_dur_avg_m0', 'lh_dur_avg_m0']] = dur_models[0]
    subj, grp = get_subject_exit_thresholds(subj, grp, sfx='_m0')

    # Exit deviations using reduced model
    #grp, summary = get_subject_exit_threshold_variation(grp, summary, time='avg', sfx='_m0')

    # Fit linear models to stay durations and exit thresholds
    #grp = fit_stay_lms_full(subj, grp, type='dur')
    #grp = fit_stay_lms_full(subj, grp, type='exit')

    # A few misc variables
    #grp['baseline_composite'] = (robust_zscore(grp['dur_avg_PC1_scores']) + robust_zscore(grp['dur_b_const']))/2
    #grp['hl_composite'] = (robust_zscore(grp['dur_avg_PC2_scores']) + robust_zscore(grp['dur_b_hl']) + robust_zscore(grp['hl_dur_dev']))/3

    return subj, grp, summary


def get_subject_policy_changes_over_time(subj, grp, robust=True):
    '''Computes linear models of subject stay durations over time'''
    
    ns    = len(subj)                       # Number of subjects
    conds = ['ac', 'hl', 'll', 'hh', 'lh']  # All conditions case is "ac"    
    endog = ['lat', 'dur', 'hgt']           # Y-variables: latencies and durations
    
    # Column names of y-variables in subject dataframes
    names = {'lat':'time_to_response', 'dur':'space_down_time', 'hgt':'height'}
    
    # Statistics to compute: intercepts, slopes, beginning, end values, averages, medians
    stats = ['b0', 'b1', 't0', 't1', 'avg', 'med']

    # Initialize fields, columns for stats for each y-variable
    for cond in conds:
        for yvar in endog:
            for stat in stats:
                grp[cond + '_' + yvar + '_' + stat] = np.NaN
    
    # For each subject, each condition, and each y-variable, mask out data and compute models
    for s in range(0, ns):
        for cond in conds:
            msk = subj[s][cond]                 # Mask for trials in this condtiion
            last_trial = grp.ntrials[s]         # Last trial number, for computing final policy

            # Compute linear models
            for yvar in endog:
                prefix = cond + '_' + yvar + '_' # Path in tree of model name strings
                if robust:
                    lm = sp.stats.siegelslopes(subj[s][names[yvar]][msk], subj[s].trial_number[msk])                
                    grp.at[s, prefix + 'b0'] = lm.intercept
                    grp.at[s, prefix + 'b1'] = lm.slope
                    grp.at[s, prefix + 't0'] = lm.intercept + lm.slope*1
                    grp.at[s, prefix + 't1'] = lm.intercept + lm.slope*last_trial
                else:
                    res = run_lm(subj[s][names[yvar]][msk], subj[s].trial_number[msk], disp = False)
                    grp.at[s, prefix + 'b0'] = res.params.const
                    grp.at[s, prefix + 'b1'] = res.params.trial_number,
                    grp.at[s, prefix + 't0'] = res.params.const + res.params.trial_number*1,
                    grp.at[s, prefix + 't1'] = res.params.const + res.params.trial_number*last_trial

                # Split halves
                shidx = int(len(subj[s][names[yvar]][msk])/2) # Split-half point
                h0 = subj[s][names[yvar]][msk][0:shidx]
                h1 = subj[s][names[yvar]][msk][shidx:]

                if len(h0) == 0 or len(h1) == 0:
                    print('Error in split halves: ' + str(s) + ' ' + cond + ' ' + prefix + str(shidx) + ' ' +str(int(last_trial)))
                else:
                    grp.at[s, prefix + 'h0'] = np.median(h0)
                    grp.at[s, prefix + 'h1'] = np.median(h1)

                # Save the averages and medians
                grp.at[s, prefix + 'avg'] = np.mean(  subj[s][names[yvar]][msk])
                grp.at[s, prefix + 'std'] = np.std(   subj[s][names[yvar]][msk])
                grp.at[s, prefix + 'med'] = np.median(subj[s][names[yvar]][msk])
                
            # Avoid fragmentation (and annoying warnings)
            if s == 0: grp = grp.copy()
    
    # Initialize reward associated with these policies
    grp['rew_t0'] = np.NaN
    grp['rew_t1'] = np.NaN
    grp['rew_dt'] = np.NaN

    # Get reward for each policy
    for s in range(0, ns):
        lats0 = np.array(grp[['hl_lat_t0', 'll_lat_t0', 'hl_lat_t0', 'll_lat_t0']].iloc[s])
        durs0 = np.array(grp[['hl_dur_t0', 'll_dur_t0', 'hl_dur_t0', 'll_dur_t0']].iloc[s])
        
        lats1 = np.array(grp[['hl_lat_t1', 'll_lat_t1', 'hl_lat_t1', 'll_lat_t1']].iloc[s])
        durs1 = np.array(grp[['hl_dur_t1', 'll_dur_t1', 'hl_dur_t1', 'll_dur_t1']].iloc[s])

        grp.at[s, 'rew_t0'] = get_policy_rew(durs0, lats0)
        grp.at[s, 'rew_t1'] = get_policy_rew(durs1, lats1)
        grp.at[s, 'rew_dt'] = grp.at[s, 'rew_t1'] - grp.at[s, 'rew_t0']
    
        grp.at[s, 'rew_pred'] = get_policy_rew(grp.dur_avg[s]*np.ones(4), grp.lat_avg[s]*np.ones(4))

    return grp





def get_policy_exit_thresholds(policy = None):
    if policy is None:
        policy = get_policy_opt()

    conds = ((0.02, 0.0002), (0.01, 0.0002), (0.02, 0.0004), (0.01, 0.0004))
    names = ['hl', 'll', 'hh', 'lh']

    # Policy input might be vector or matrix
    if len(policy.shape) == 1:
        policy = policy.reshape(1,4)
    
    if isinstance(policy, pd.DataFrame):
        policy = policy.values

    # These should all be the same number, if using optimal policy
    out = np.zeros_like(policy)
    for i, (name, cond) in enumerate(zip(names,conds)):
        out[:,i] = cond[0] * np.exp(-cond[1]*policy[:,i])

    return out.squeeze()


def get_policy_from_exit_thresholds(thresholds):
    conds = ((0.02, 0.0002), (0.01, 0.0002), (0.02, 0.0004), (0.01, 0.0004))
    names = ['hl', 'll', 'hh', 'lh']

    # These should all be the same number, if using optimal policy
    policy = np.zeros_like(thresholds)
    for i, (name, cond) in enumerate(zip(names,conds)):
        policy[:,i] = np.log(thresholds[:,i] / cond[0]) / -cond[1]

    return policy.squeeze()


def get_subject_exit_thresholds(subj, grp, time='avg', sfx='', ftest=False):

    # Patch params (base, decay) for dm/dt = base * exp(-decay * time)
    conds = ((0.02, 0.0002), (0.01, 0.0002), (0.02, 0.0004), (0.01, 0.0004))
    names = ['hl', 'll', 'hh', 'lh']

    for name, cond in zip(names,conds):
        grp[name+'_exit_'+time+sfx] = cond[0] * np.exp(-cond[1]* grp[name+'_dur_'+time+sfx])
        #grp[name+'_exit_ub' ] = cond[0] * np.exp(-cond[1]*(grp[name+'_dur_avg'+sfx] + 1*grp[name+'_dur_std']))
        #grp[name+'_exit_lb' ] = cond[0] * np.exp(-cond[1]*(grp[name+'_dur_avg'+sfx] - 1*grp[name+'_dur_std']))

    for i, s in enumerate(subj):
        subj[i]['exit'] = s['S']*np.exp(-s['decay_rate']*s['space_down_time'])

    if ftest:
        grp['exit_ftest_pval'] = 1

        for i, s in enumerate(subj):

            subj[i].loc[subj[i]['hl'], 'type'] = 2
            subj[i].loc[subj[i]['ll'], 'type'] = 1
            subj[i].loc[subj[i]['hh'], 'type'] = -1
            subj[i].loc[subj[i]['lh'], 'type'] = -2

            model = ols('exit ~ type', data = subj[i]).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            #print(anova_table['PR(>F)'][0])
            
            grp['exit_ftest_pval'][i] = anova_table['PR(>F)'][0]

    # Exit means, sqrt(means), log(means)
    exits = np.array(grp[ ['hl_exit_'+time, 'll_exit_'+time, 'hh_exit_'+time, 'lh_exit_'+time] ])
    grp['exit_means_'+time] = np.mean(exits,axis=1)
    grp['sqrt_exit_means_'+time] = np.sqrt(grp['exit_means_'+time])
    grp['log_exit_means_'+time] = np.log(grp['exit_means_'+time])

    return subj, grp

def get_subject_exit_threshold_variation(grp, summary, time='avg', sfx=''):
    cols = ['hl_exit', 'll_exit', 'hh_exit', 'lh_exit']
    cols = ['_'.join([c,time])+sfx for c in cols] 
    
    cols = ['hl_exit_'+time+sfx, 'll_exit_'+time+sfx, 'hh_exit_'+time+sfx, 'lh_exit_'+time+sfx]
    exit_avg = grp[cols].values

    devs  = (exit_avg - np.mean(exit_avg,axis=1).reshape(-1,1))
    diffs = (np.max(devs, axis=1) - np.min(devs,axis=1))

    cols = ['_'.join([c,'dev']) for c in cols] 

    grp[cols] = devs
    grp['exit_delta_'+time+sfx] = diffs


    summary['exit_dev_means_'+time+sfx] = np.mean(devs, axis=0)
    summary['exit_dev_sems_'+time+sfx]  = sp.stats.sem(devs, axis=0)
    return grp, summary


def get_subject_dur_deviations(grp):
    '''Computes deviation in stay durations from those predicted by subject average threshold'''
    # Get subject average exit thresholds
    cols = ['hl_exit_avg', 'll_exit_avg', 'hh_exit_avg', 'lh_exit_avg']
    exit_avg = np.mean(grp[cols].values,axis=1).reshape(-1,1)

    # Convert these to a policy
    policy_avg = get_policy_from_exit_thresholds(np.tile(exit_avg,[1,4]))

    # Compute deviations from policy
    devs = (grp[['hl_dur_avg', 'll_dur_avg', 'hh_dur_avg', 'lh_dur_avg']].values - policy_avg)
    diffs = (np.max(devs, axis=1) - np.min(devs,axis=1))

    grp[['hl_dur_dev', 'll_dur_dev','hh_dur_dev', 'lh_dur_dev']] = devs
    grp['dur_delta'] = diffs

    return grp



def get_policies(grp):
    flds = ['hl_dur_avg', 'll_dur_avg', 'hh_dur_avg', 'lh_dur_avg']
    avg  = grp[flds]
    return avg


def get_reduced_model_rve(grp, type='dur'):
    # Latencies
    lats = np.array(grp[ ['hl_lat_avg', 'll_lat_avg', 'hh_lat_avg', 'lh_lat_avg'] ])

    if type == 'dur':
        # Average stay durations
        vals = np.array(grp[ ['hl_dur_avg', 'll_dur_avg', 'hh_dur_avg', 'lh_dur_avg'] ])

    elif type == 'exit':
        # Exit thresholds
        vals = np.array(grp[ ['hl_exit_avg', 'll_exit_avg', 'hh_exit_avg', 'lh_exit_avg'] ])

    # PCA of policy variables
    evals, evecs = np.linalg.eig(np.cov(vals.T))
        
    # Iterable of models
    models = [evecs[:,0:1], evecs[:,0:2], evecs[:,0:3], evecs[:,0:4]]
    
    # Init model stats lists
    model_rve, model_beta, model_proj = [], [], []

    # Get statistics
    for m, model in enumerate(models):

        # Get betas
        model_beta.append( np.matmul(vals, model) )
        
        # Subjects' policy models
        model_proj.append( np.matmul(model_beta[m], model.T) )
        
        # Convert to policy if using exits
        if type == 'exit':
            model_proj[m] = get_policy_from_exit_thresholds(model_proj[m])

        # Reward variance explained
        model_rve.append( get_policy_rve(model_proj[m], lats, grp.rew_tot) )
    
    return model_rve, model_proj
    



    
def get_policy_ve(subspace, data):
    
    # Rejection matrix for model
    rejection = np.eye(4) - np.matmul(subspace, subspace.T)
    
    # Residual error
    res = np.matmul(rejection, data.T).T
    
    # Sum of squared errors and total sum of squares
    sse = np.sum((res -  np.mean(res ,0))**2)
    sst = np.sum((data - np.mean(data,0))**2)
    
    # Variance explained by the model
    VE = 1 - sse/sst
    
    return VE


def get_policy_rve(proj, lats, rew):
    
    # Number of subjects
    ns = np.shape(lats)[0]
    
    # Reward predicted from mean stay durations
    rew_pred = np.zeros(ns)
    for s in range(0, ns):
        rew_pred[s] = get_policy_rew(proj[s], lats[s,:])
    
    # Unexplained reward
    res = rew - rew_pred
    
    # Sums of squares
    sse = np.sum((res - np.mean(res, 0))**2)
    sst = np.sum((rew - np.mean(rew, 0))**2)

    # Reward variance explained
    rve = 1 - sse/sst
    
    return rve


# Compute policy heat map
def run_policy_sweep(deltas, scales):
    # Base changes (deltas) and mod factors (scales)
    
    # Policy and experiment parameters
    latency  = [500, 500, 500, 500]

    # Optimal policy
    policy_opt = get_policy_opt()
    
    # Decompose into mean (baseline) duration and modulation by patch
    base = np.mean(policy_opt)
    mod  = policy_opt - base
    
    # Sweep over policies changing main rate and modulation
    rewards = np.zeros([len(deltas), len(scales)])
    for i, delta in enumerate(deltas):
        for j, scale in enumerate(scales):
        
            # New policy
            policy = base + delta + scale * mod
            
            # Threshold at zero
            policy[policy < 0] = 0
                        
            # Policy results
            total = get_policy_rew(policy, latency)
            
            # Save total reward
            rewards[i,j] = total

    #
    return rewards


# Find optimal policy via optimization
def get_policy_opt():
    # Initial patch durations guess [ms]
    # Setting to opt can mask bugs introduced by refactoring.
    initial = [6000, 5000, 4500, 3500]
    
    # Bounds
    bnds = ((1e3, 1.5e4), (1e3, 1.5e4), (1e3, 1.5e4), (1e3, 1.5e4))
    
    # Call optimziation
    result = sp.optimize.minimize(policy_wrapper, initial, bounds = bnds)

    # Return optimal policy
    return result['x']


def policy_wrapper(duration):
    # Need to return the inverse of reward (for minimization)
    reward = get_policy_rew(duration)
    
    return -1.0 * reward


# Expected patch visits and rewards given a policy
def get_policy_rew(duration, latency = [650, 650, 650, 650]):
    '''
    Inputs:
        conds    - a tuple pairs (base, decay), with decay units [1/ms]
        travel   - inter-patch travel time [ms]
        latency  - latencies before initiation of foraging in each patch [ms]
        duration - foraging durations for each patch [ms]
        time_exp - the total time in the experiment [ms]
    
    Outputs:
        repeats - expected number of cycles through patches
        rews    - reward collected in each patch type
        total   - expected total reward
    '''   
    # Task parameters
    travel   = 4e3    # Inter-patch travel time (switch cost) in [ms]
    time_exp = 600e3  # Experiment length in [ms]

    # Patch params (base, decay) for dm/dt = base * exp(-decay * time)
    conds = ((0.02, 0.0002), (0.01, 0.0002), (0.02, 0.0004), (0.01, 0.0004))

    # Number of conditions
    nconds = len(conds)

    # Compute the recurrence time for cycling through all patches
    time_rec = 4*travel + sum(latency + duration)
    
    # Expected number of patch cycles
    repeats = time_exp / time_rec
    
    # Rewards from each patch
    rews = np.zeros(nconds)
    for i in range(0, nconds):
        
        # Integrated patch reward: base / decay * (1 - exp( -decay * time ) )
        rews[i] = 1/100 * conds[i][0] / conds[i][1] * (1 - np.exp(-conds[i][1] * duration[i]))
    
    # Total reward
    total = sum(rews) * repeats
    
    # Number of trials
    # ntrials = repeats*4
    
    # Give back number of cycles and total reward
    return total #, ntrials


# Relation of a policy to the optimal, in RMSE
def get_policy_rmse(policy, opt_policy):
    diff = np.squeeze(opt_policy) - np.squeeze(policy)
    rmse = np.sqrt(sum(diff*diff))

    return rmse


# Determine where subjects live in base-vs-mod space
def policy_space(policy_opt, grp):
    
    # Baseline and modulation
    base = np.mean(policy_opt)
    mod  = policy_opt - base

    # Conditions
    conds = ['hl_dur_avg', 'll_dur_avg', 'hh_dur_avg', 'lh_dur_avg']
    
    # Get subject coords
    ns = np.shape(grp)[0]
    subj_base  = np.zeros(ns)
    subj_delta = np.zeros(ns)
    subj_mod   = np.zeros(ns)
    for i in range(0, ns):
        
        # Final policy
        policy = grp.iloc[i,:][conds]
        
        # Subject baseline
        subj_base[i]  = np.mean(policy)
        subj_delta[i] = subj_base[i] - base
        
        # Subject deviation
        diff = (policy - subj_base[i])
        
        # Modulation projection
        scale = np.linalg.norm(mod)
        subj_mod[i]  = np.dot(diff/scale, mod/scale)
        
    return subj_base, subj_delta, subj_mod


# Try to predict reward based on variability
def run_variance_lms(grp):
    
    # Correlations between potential predictors
    np.corrcoef(grp.dur_avg, grp.lat_avg)
    
    np.corrcoef(grp.lat_avg, grp.lat_std)
    np.corrcoef(grp.dur_avg, grp.dur_std)
    
    np.corrcoef(grp.dur_std, grp.lat_std)
    
    
    # Direct prediction of reward using variance
    X = grp[['lat_std', 'dur_std']]
    X = sp.stats.zscore(X)
    y = sp.stats.zscore(grp.rew_tot)
    
    mod = sm.OLS(y, X)
    
    res = mod.fit()
    
    print(res.summary())


    # Prediction using residualized variables
    X = grp[['lat_avg', 'dur_avg']]
    X = sp.stats.zscore(X)
    
    y1 = sp.stats.zscore(grp.lat_std)
    y2 = sp.stats.zscore(grp.dur_std)
    y3 = sp.stats.zscore(grp.rew_tot)
    
    mod1 = sm.OLS(y1, X)
    mod2 = sm.OLS(y2, X)
    
    res1 = mod1.fit()
    res2 = mod2.fit()
    
    print(res1.summary())
    print(res2.summary())
    
    X['lat_res'] = res1.resid
    X['dur_res'] = res2.resid
    X = sm.add_constant(X)
    
    mod3 = sm.OLS(y3, X)
    res3 = mod3.fit()
    
    print(res3.summary())
    
    



def get_noham_AES_stay_rho_1(subj, grp):
    
    # Number of subjects
    ns = len(subj)
    
    # Beta lists (to frame later)
    b0, b1, b2 = [], [], []
    
    # Loop through subjects getting their betas
    for s in range(0, ns):
        
        # Get subject duration data
        X = subj[s][['hl','ll','hh','lh']]
    
        # Boolean predictor columns for init and decay
        X2 = pd.DataFrame( [X['hh'] + X['hl'], X['hh'] + X['lh']], index=['h-', '-h']).T
        X2 = X2.applymap(lambda x: 1 if x == True else 0)
    
        # Replace booleans with init and decay parameter values
        X2['h-'].map(lambda x: 0.02   if x == 1 else 0.01  )
        X2['-h'].map(lambda x: 0.0004 if x == 1 else 0.0002)
        
        # Add intercept to model
        X2 = sm.add_constant(X2)
    
        # Generate and fit
        mod = sm.OLS(subj[s].space_down_time, X2)
        res = mod.fit()
        
        # Save the beta values
        b0.append(res.params.const)
        b1.append(res.params['h-'])
        b2.append(res.params['-h'])


    # Concatenate betas into a frame
    betas = pd.DataFrame([b0,b1,b2], index=['b0', 'b1', 'b2']).T
    
    # Predict AES with betas (linear model)
    #res = run_lm(grp.aes, betas, zscore = True, robust = True)

    # Correlate AES rank with beta 0
    res = sp.stats.spearmanr(grp.aes, b0)

    print('\nSpearman rho (AES, Stay-Baseline):')
    print(res.statistic)
    
    print('\np-value:')
    print(res.pvalue)

    return betas


def get_noham_AES_stay_rho_2(subj, grp):
    
    # Number of subjects
    ns = len(subj)
    
    # Beta lists (to frame later)
    b0, b1, b2 = [], [], []
    
    # Loop through subjects getting their betas
    for s in range(0, ns):
        
        # Get subject duration data
        X = subj[s][['S', 'decay_rate']]
    
        # Add intercept to model
        X = sm.add_constant(X)
    
        # Generate and fit
        mod = sm.RLM(subj[s].space_down_time, X)
        res = mod.fit()
        
        # Save the beta values
        b0.append(res.params.const)
        b1.append(res.params['S'])
        b2.append(res.params['decay_rate'])

    # Concatenate betas into a frame
    betas = pd.DataFrame([b0,b1,b2], index=['b0', 'b1', 'b2']).T
    
    # Predict AES with betas (linear model)
    #res = run_lm(grp.aes_rank, betas, zscore = False, robust = True)

    # Correlate AES rank with beta 0
    res = sp.stats.spearmanr(grp.aes, b0)
        
    print('\nSpearman rho (AES, Stay-Baseline):')
    #print(res.statistic)
    print(res)
    
    print('\np-value:')
    print(res.pvalue)

    return betas


def get_noham_AES_stay_rho_3(subj, grp):
    
    # Number of subjects
    ns = len(subj)
    
    # Beta lists (to frame later)
    b0, b1, b2 = [], [], []
    
    # Loop through subjects getting their betas
    for s in range(0, ns):
        
        # Get subject duration data
        X = subj[s][['S', 'decay_rate']]
        X = sp.stats.zscore(X)
    
        # Add intercept to model
        X = sm.add_constant(X)
        
        # Outcome variable
        #y = sp.stats.zscore(subj[s].space_down_time)
        y =  subj[s].space_down_time
        
        # Generate and fit
        mod = sm.RLM(y, X)
        res = mod.fit()
        
        # Save the beta values
        b0.append(res.params.const)
        b1.append(res.params['S'])
        b2.append(res.params['decay_rate'])

    # Concatenate betas into a frame
    betas = pd.DataFrame([b0,b1,b2], index=['b0', 'b1', 'b2']).T
    
    # Predict AES with betas (linear model)
    # res = run_lm(grp.aes, betas, zscore = True, robust = True)

    # Correlate AES rank with beta 0
    res = sp.stats.spearmanr(grp.aes, b0)
        
    print('\nSpearman rho (AES, Stay-Baseline):')
    print(res.statistic)
    
    print('\np-value:')
    print(res.pvalue)

    return betas


def get_noham_AES_stay_rho_4(subj, grp):
    
    # Number of subjects
    ns = len(subj)
    
    # Beta lists (to frame later)
    b0, b1, b2 = [], [], []
    
    # Loop through subjects getting their betas
    for s in range(0, ns):
        
        # Get subject duration data
        X = subj[s][['S', 'decay_rate']]
        X = sp.stats.zscore(X)
    
        # Add intercept to model
        X = sm.add_constant(X)
        
        # Outcome variable
        #y = sp.stats.zscore(subj[s].space_down_time)
        
        y = robust_zscore(subj[s].space_down_time)
        
        # Generate and fit
        mod = sm.RLM(y, X)
        res = mod.fit()
        
        # Save the beta values
        b0.append(res.params.const)
        b1.append(res.params['S'])
        b2.append(res.params['decay_rate'])

    # Concatenate betas into a frame
    betas = pd.DataFrame([b0,b1,b2], index=['b0', 'b1', 'b2']).T
    
    # Predict AES with betas (linear model)
    # res = run_lm(grp.aes, betas, zscore = True, robust = True)

    # Correlate AES rank with beta 0
    res = sp.stats.spearmanr(grp.aes, b0)
        
    print('\nSpearman rho (AES, Stay-Baseline):')
    print(res.statistic)
    
    print('\np-value:')
    print(res.pvalue)

    return betas



def fit_stay_lms_marginal(subj):
    
    # Number of subjects
    ns = len(subj)
    
    # Beta lists (to frame later)
    results  = pd.DataFrame(np.zeros([ns,8]), columns = ['b_const','b_h-','b_-h', 'p_const','p_h-','p_-h', 'r2', 'fp'])

    # Loop through subjects getting their betas
    for s in range(0, ns):
        
        # Get subject duration data
        X = subj[s][['hl','ll','hh','lh']]
    
        # Boolean predictor columns for init and decay
        X2 = pd.DataFrame( [X['hh'] + X['hl'], X['hh'] + X['lh']], index=['h-', '-h']).T
        X2 = X2.applymap(lambda x: 1 if x == True else 0)
    
        # Add intercept to model
        X2 = sm.add_constant(X2)
    
        # Generate and fit
        mod = sm.OLS(subj[s].space_down_time, X2)
        res = mod.fit()

        # Concatenate betas into a frame
        results.iloc[s,0:3] = res.params.values
        results.iloc[s,3:6] = res.pvalues.values
        results.iloc[s,6]   = res.rsquared
        results.iloc[s,7]   = res.f_pvalue

    return results


def fit_stay_lms_full(subj, grp, type='dur'):
    
    # Number of subjects
    ns = len(subj)
    
    # Beta lists (to frame later)
    cols = ['b_const','b_h-','b_-h','b_hl', 'p_const','p_h-','p_-h','p_hl', 'r2', 'fp']
    cols = [type + '_' + col for col in cols]
    results  = pd.DataFrame(np.zeros([ns,10]), columns = cols)

    # Loop through subjects getting their betas
    for s in range(0, ns):
        
        # Get subject duration data
        X = subj[s][['hl','ll','hh','lh']]
    
        # Boolean predictor columns for init and decay
        X2 = pd.DataFrame( [X['hh'] + X['hl'], X['hh'] + X['lh']], index=['h-', '-h']).T
        X2 = X2.applymap(lambda x: 1 if x == True else 0)
        #X3 = X['hh'].apply(lambda x: 1 if x else 0) + X['ll'].apply(lambda x: -1 if x else 0)
        X3 = X['hl'].apply(lambda x: 1 if x else 0)
        X3.name = 'hl'
        X2 = X2.join(X3)
    
        # Add intercept to model
        X2 = sm.add_constant(X2)
    
        # Generate and fit
        if type == 'dur':
            y = subj[s].space_down_time
        elif type == 'exit':
            y = np.sqrt(subj[s].exit)

        mod = sm.OLS(y, X2)
        res = mod.fit()

        # Concatenate betas into a frame
        results.iloc[s,0:4] = res.params.values
        results.iloc[s,4:8] = res.pvalues.values
        results.iloc[s,8]   = res.rsquared
        results.iloc[s,9]   = res.f_pvalue

    grp = grp.join(results)

    return grp


    

def get_stay_pca(grp, summary, time='avg', var='dur', verbose=False):
    if verbose:
        print('Getting PCA for ' + var + ' at ' + time)
    
    vals = np.array(grp[ ['hl_'+var+'_'+time, 'll_'+var+'_'+time, 'hh_'+var+'_'+time, 'lh_'+var+'_'+time] ])

    # PCA itself, jackknife standard errors
    evals, evecs, scores = PCA(vals, pos_const=True)
    evals_jse = jse(vals, lambda x: PCA(x, pos_const=True)[1])

    scale = {'dur':1/10000, 'exit':1000, 'hgt':1}

    grp[var+'_'+time+'_pc1_score'] = (vals - np.mean(vals,axis=0))@ evecs[:,0]*scale[var]
    grp[var+'_'+time+'_pc2_score'] = (vals - np.mean(vals,axis=0))@ evecs[:,1]*scale[var]

    if var == 'dur':
        grp['dur_baseline']   = (vals - np.mean(vals,axis=0))@ np.ones(4)/np.sqrt(4)*scale[var]

    for i in range(0,4):
        summary[var+'_'+time+'_pc'+str(i+1)] = evecs[:,i]
        summary[var+'_'+time+'_pc'+str(i+1)+'_jse'] = evals_jse[:,i]

    return grp, summary



def run_age_lms(endog, age):
    endog_cols = endog.columns.tolist()

    # Linear regressions for exogenous variables by age
    models = pd.DataFrame(np.zeros([4,len(endog_cols)]), ['standard_coef', 'standard_err', 'p-value', 'r-squared'], columns = endog_cols)
    for col in endog_cols:
        
        # Get LMs for predicting each exogenous variable by age
        res = run_lm(endog[col], age, zscore = True, robust = False, add_const = True, disp=True)

        # Collect results (we'll plot them below)
        models.loc['standard_coef',col] = res.params[1]
        models.loc['standard_err', col] = res.bse[1]
        models.loc['p-value'      ,col] = res.pvalues[1]
        models.loc['r-squared'    ,col] = res.rsquared

        # Residualized
        endog[col] = get_resid(endog[col], age, disp = False)

    return endog, models

def run_lms(endog, exog):
    models = {}
    cnames = exog.columns.tolist()
    for col in endog.columns.tolist():
        X = exog.copy()
        models[col] = pd.DataFrame(np.zeros([3,len(cnames)]), ['standard_coef', 'standard_err', 'p-value',], columns = cnames)

        # Drop predictor column
        if col in X.columns:
            X = X.drop(col, axis=1)

        # Linear model
        res = run_lm(endog[col], X, zscore = True, robust = False, add_const = True, disp=True)

        # Collect results (we'll plot them below)
        models[col].loc['standard_coef'] = res.params[1:]
        models[col].loc['standard_err' ] = res.bse[1:]
        models[col].loc['p-value'      ] = res.pvalues[1:]
    
    return models

