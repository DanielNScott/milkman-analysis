#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:59:27 2023

@author: dan
"""
import numpy  as np
import pandas as pd
import scipy  as sp

import statsmodels.api as sm

def analyze_subjects(subj, grp):
    
    # Number of subjects
    ns = len(subj)
    
    # 
    grp['rew_tot'] = [subj[i].total_milk.iloc[-1]         for i in range(0,ns)]
    
    grp['lat_avg'] = [np.mean(subj[i].time_to_response)   for i in range(0,ns)]
    grp['dur_avg'] = [np.mean(subj[i].space_down_time)    for i in range(0,ns)]
    grp['hgt_avg'] = [np.mean(subj[i].height)             for i in range(0,ns)]
    
    grp['lat_med'] = [np.median(subj[i].time_to_response) for i in range(0,ns)]
    grp['dur_med'] = [np.median(subj[i].space_down_time)  for i in range(0,ns)]
    grp['hgt_med'] = [np.median(subj[i].height)           for i in range(0,ns)]
    
    grp['lat_std'] = [np.std(subj[i].time_to_response)    for i in range(0,ns)]
    grp['dur_std'] = [np.std(subj[i].space_down_time)     for i in range(0,ns)]
    grp['hgt_std'] = [np.std(subj[i].height)              for i in range(0,ns)]
    
    grp['ntrials'] = [subj[i].trial_number.iloc[-1]       for i in range(0,ns)]
    
    # Get some subject rank orders
    grp['rew_rank'] = get_rank(grp.rew_tot)
    grp['aes_rank'] = get_rank(grp.aes)
    grp['dur_rank'] = get_rank(grp.dur_avg)
    
    # Defrag to avoid annoying warnings
    frame = grp.copy()
    del grp
    grp = frame.copy()
    del frame
    
    # Get linear models of subject timing data
    grp = get_policy_stats(subj, grp)

    # Get policies
    grp['rew_t0'] = np.NaN
    grp['rew_t1'] = np.NaN
    grp['rew_dt'] = np.NaN
    for s in range(0, ns):
        lats0 = np.array(grp[['hl_lat_t0', 'll_lat_t0', 'hl_lat_t0', 'll_lat_t0']].iloc[s])
        durs0 = np.array(grp[['hl_dur_t0', 'll_dur_t0', 'hl_dur_t0', 'll_dur_t0']].iloc[s])
        
        lats1 = np.array(grp[['hl_lat_t1', 'll_lat_t1', 'hl_lat_t1', 'll_lat_t1']].iloc[s])
        durs1 = np.array(grp[['hl_dur_t1', 'll_dur_t1', 'hl_dur_t1', 'll_dur_t1']].iloc[s])

        grp.at[s, 'rew_t0'] = get_policy_rew(durs0, lats0)
        grp.at[s, 'rew_t1'] = get_policy_rew(durs1, lats1)
        grp.at[s, 'rew_dt'] = grp.at[s, 'rew_t1'] - grp.at[s, 'rew_t0']
    
        grp.at[s, 'rew_pred'] = get_policy_rew(grp.dur_avg[s]*np.ones(4), grp.lat_avg[s]*np.ones(4))
    
    # Get residualized trialwise variabilies
    
    # Prediction using residualized variables
    grp['lat_std_res'] = np.NaN
    grp['lat_std_res'] = get_resid(grp.lat_std, grp.lat_avg, disp = False)
    
    grp['dur_std_res'] = np.NaN
    grp['dur_std_res'] = get_resid(grp.dur_std, grp.dur_avg, disp = False)
    
    grp['hgt_std_res'] = np.NaN
    grp['hgt_std_res'] = get_resid(grp.hgt_std, grp.hgt_avg, disp = False)

    return grp


def get_rank(col):
    ns  = len(col)
    idx = np.argsort(col)
    
    subj = np.arange(0, ns)
    rank = np.arange(0, ns)
    
    subj[idx] = rank
    
    return subj


def get_policy_stats(subj, grp):
    
    # Number of subjects
    ns = len(subj)
    
    # Conditions from best to worst case, as usual, plus "all conditions"
    conds = ['ac', 'hl', 'll', 'hh', 'lh']
    
    # Values: model latencies and durations
    vals = ['lat', 'dur', 'hgt']
    vdic = {'lat':'time_to_response', 'dur':'space_down_time', 'hgt':'height'}
    
    # Model results: intercepts, slopes, beginning, end values
    mres = ['b0', 'b1', 't0', 't1', 'avg', 'med']

    # Columns for linear model betas, initial and final values, by condition
    # Initialize fields
    for cond in conds:
        for val in vals:
            for res in mres:
                grp[cond + '_' + val + '_' + res] = np.NaN
    
    # Loop over subjects
    for s in range(0, ns):
        
        # Loop over conditions
        for cond in conds:
            # Trial type
            msk = subj[s][cond]
            
            # Get last trial for computing final policies
            last_trial = grp.ntrials[s]
            
            # Compute models on both latency and duration
            for val in vals:
                
                # Path in tree of model name strings
                prefix  = cond + '_' + val + '_'
                
                # Robust regression of stay-durations on trial numbers
                lm = sp.stats.siegelslopes(subj[s][vdic[val]][msk], subj[s].trial_number[msk])
                
                # Save the linear model
                grp.at[s, prefix + 'b0'] = lm.intercept
                grp.at[s, prefix + 'b1'] = lm.slope
                grp.at[s, prefix + 't0'] = lm.intercept + lm.slope*1
                grp.at[s, prefix + 't1'] = lm.intercept + lm.slope*last_trial
    
                # Save the averages and medians
                grp.at[s, prefix + 'avg'] = np.mean(  subj[s][vdic[val]][msk])
                grp.at[s, prefix + 'std'] = np.std(   subj[s][vdic[val]][msk])
                grp.at[s, prefix + 'med'] = np.median(subj[s][vdic[val]][msk])
                
            # Avoid fragmentation (and annoying warnings)
            if s == 0: grp = grp.copy()
    
    return grp


def get_policies(grp):
    flds = ['hl_dur_avg', 'll_dur_avg', 'hh_dur_avg', 'lh_dur_avg']
    avg  = grp[flds]    
    return avg


def get_policy_models(subj, grp):
    
    # Average stay durations
    durs = np.array(grp[ ['hl_dur_avg', 'll_dur_avg', 'hh_dur_avg', 'lh_dur_avg'] ])
    lats = np.array(grp[ ['hl_lat_avg', 'll_lat_avg', 'hh_lat_avg', 'lh_lat_avg'] ])

    # Get the optimal policy
    policy_opt  = get_policy_opt()
    
    # Define constant and mean-zero policy components
    policy_cnst = np.ones(4)/np.sqrt(4)
    
    policy_mzro = policy_opt - np.mean(policy_opt)
    policy_mzro = policy_mzro/np.linalg.norm(policy_mzro)
    
    # PCA of policy variables
    evals, evecs = np.linalg.eig(np.cov(durs.T))
    
    # fig = figure()
    # plot(vals/sum(evals), '-o')
    # ylim([0,1])
    # grid('on')
    # ylabel('Fraction Variance Explained')
    # xlabel('Principal Component')
    # title('PCA Variance Explained')
    # fig.savefig('./response-var-explained.png')
    
    # Subspace 0, full model (correctness check & reward VE ceiling)
    S0 = evecs

    # Subspace 1, top 2-eigenvector space
    S1 = np.vstack([evecs[:,0], evecs[:,1]]).T
    
    # Subspace 2, policy-decomposition
    S2 = np.vstack([policy_cnst, policy_mzro]).T
    
    # Subspace 3, pure baseline model
    S3 = np.array([S2[:,0]]).T
    
    # Determine angles between subspaces
    angles_rad = sp.linalg.subspace_angles(S1, S2)
    angles_deg = np.rad2deg(angles_rad)
    
    # Convert to rho-values
    rho_S1_S2 = np.cos(angles_rad)
    
    # Iterable of models
    model = [S0, S1, S2, S3]
    
    # Init model stats lists
    model_ve, model_rve, model_beta, model_proj = [], [], [], []

    # Get statistics
    for m in range(0,4):
    
        # Get policy variance explained by models
        model_ve.append( get_policy_ve(model[m], durs) )
    
        # Get betas
        model_beta.append( np.matmul(durs, model[m]) )
        
        # Subjects' policy models
        model_proj.append( np.matmul(model_beta[m], model[m].T) )
        
        # Reward variance explained
        model_rve.append( get_policy_rve(model_proj[m], lats, grp.rew_tot) )
    
    # Print angles and rhos, variances explained, etc
    print('Subspace angles [deg]:')
    print(angles_deg)
    
    print('\nSubspace angles [rad]:')
    print(angles_rad)
    
    print('\nSubspace rho-values:')
    print(rho_S1_S2)
    
    print('\nPolicy variance explained:')
    print(model_ve)
    
    print('\nReward variance explained:')
    print(model_rve)
    
    return model_ve, model_rve, model_beta, model_proj
    
    
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
    res = rew- rew_pred
    
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
    
    
# Wrapper for residualizing variables
def get_resid(y, X, disp = True):
    
    # 
    X = sm.add_constant(X)
        
    mod = sm.OLS(y, X)
    res = mod.fit()
    
    if disp:
        print(res.summary())
    
    return res.resid


# Wrapper for the scipy linear model call
def run_lm(y, X, disp = True, zscore = False, robust = False):
    
    # Z-score data
    if zscore:
        X = sp.stats.zscore(X)
        y = sp.stats.zscore(y)
    else:
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


def get_noham_AES_stay_rho(subj, grp):
    
    # Number of subjects
    ns = len(subj)
    
    # Beta lists (to frame later)
    b0 = []
    b1 = []
    b2 = []
    
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
        mod = sm.RLM(subj[s].space_down_time, X2)
        res = mod.fit()
    
        # Save the beta values
        b0.append(res.params.const)
        b1.append(res.params['h-'])
        b2.append(res.params['-h'])
    
    # Concatenate betas into a frame
    betas = pd.DataFrame([b0,b1,b2], index=['b0', 'b1', 'b2']).T
    
    # Predict AES with betas (linear model)
    res = run_lm(grp.aes, betas)
    
    # Correlate AES rank with beta 0
    res = sp.stats.spearmanr(grp.aes, b0)
    
    print('\nSpearman rho (AES, Stay-Baseline):')
    print(res.statistic)
    
    print('\np-value:')
    print(res.pvalue)

    #return res.params.const