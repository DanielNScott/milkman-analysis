#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:16:52 2023

@author: dan
"""
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

# Imports from my modules
#from utils import stats, mad
from analy import run_policy_sweep, policy_space, get_policy_opt, get_policies


def plot_all(ns, save, dir_plots, subj, cond):

    # Determine which plots exist
    plots_subj_level  = [i for i in dir_plots if 'plot_subj_'  in i]
    plots_group_level = [i for i in dir_plots if 'plot_group_' in i]
    
    # Get and format current date and time, make a parent figure directory
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = './data/figs/' + now_str + '/'
    mkdir(path)
    
    # Turn interactive plotting off while generating and saving figures
    #ioff()
    
    # Cycle through subjects and plot things
    for i in range(0,ns):
        #path = './data/figs/'+ now_str +'/subj-' + str(i).zfill(2) + '/'
        #mkdir(path)
        
        # Plot all subject level plots
        args =  'subj[' + str(i) + '], cond[' + str(i) + '], ' + str(i) + ', ' + str(save) + ', path'
        for j in plots_subj_level:
            exec(j +'(' + args + ')')
    
    # Make group level plots
    args =  'subj, ' + str(save) + ', path'
    for j in plots_group_level:
        exec(j +'(' + args + ')')
    
            
    # Turn interactive plotting back on
    #ion()



def plot_trial_timing(subj, grp, snum, save = False, path = ''):
    
    # Var should be either 'lat' or 'dur'
    vdict = {'lat':'time_to_response' , 'dur':'space_down_time'   ,  'hgt':'height'}
    xdict = {'lat':'Latency [s]'      , 'dur':'Stay Duration [s]' ,  'hgt':'Height [%]'}
    sdict = {'lat':1e-3               , 'dur':1e-3                ,  'hgt':1}
    
    lbds  = {'lat':0.2                , 'dur':2                   ,  'hgt':0  }
    ubds  = {'lat':1.2                , 'dur':18                  ,  'hgt':100}
    incs  = {'lat':0.2                , 'dur':2                   ,  'hgt':10}
    
    # New figure    
    fig = figure(figsize = [15,9])
    
    # Get color cycle to match with lines
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Long-form condition labels
    clabels = ['All', 'High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']

    # Plot both latencies and durations
    for row, var in enumerate(['lat', 'dur', 'hgt']):
        
        # Get scale ([ms] to [s] for first 2)
        scale = sdict[var]
    
        # Plot each condition    
        for col, cond in enumerate(['ac', 'hl', 'll', 'hh', 'lh']):
            
            # Next subplot
            subplot(3, 5, row*5 + col + 1)
            
            # Data
            trls = subj[snum].loc[subj[snum][cond], 'trial_number']
            yval = subj[snum].loc[subj[snum][cond], vdict[var]] * scale
            
            # Plot data
            plot(trls, yval,'o')
                    
            # Linear model endpoints
            x  = [1, grp['ntrials'].iloc[snum]]
            y  = [grp[cond + '_' + var + '_t0'].iloc[snum] * scale,
                  grp[cond + '_' + var + '_t1'].iloc[snum] * scale] 
            
            # Plot linear model
            plot(x, y, colors[col], linestyle = 'dashed')
            
            # Set y-axes
            #if cond != 'ac':
            yticks( np.arange(lbds[var], ubds[var] + incs[var], incs[var]) )
            ylim([lbds[var], ubds[var]])

            # Label
            label('Trial Number', xdict[var], clabels[col])

    # Save if flag set
    save_wrapper(fig, save, path, 'trial-vars-', snum)




## ---------------------------------------------- Group Level ----------------------------------------------- ##
def plot_policy_models(model_ve, model_rve):
    figure(figsize = [3,3])
    plt.bar(range(0,3), model_ve[1:4])    
    plt.bar(range(3,6), model_rve[1:4])

    xticks(range(0,6), labels = ['M1', 'M2', 'M3', 'M1', 'M2', 'M3'])
    xlabel('Models')
    legend(['Policy','Reward'])
    ylabel('Fraction Variance Explained')
    title('Model Comparison')
    
    tight_layout()

def plot_policy_rew_deltas(grp):
    figure(figsize = [3,3])
    
    rew_delta = grp.rew_t1 - grp.rew_t0
    scatter(grp.ac_dur_b1, rew_delta, c = grp.rew_tot)
    
    title('Policy Change Impacts')
    xlabel('$\Delta$ Duration [ms per trial]')
    ylabel('$\Delta$ Reward')
    tight_layout()
    
    res = sp.stats.linregress(grp.ac_dur_b1, grp.rew_dt)
    
    print('\nRegression r-value:')
    print(res.rvalue)
    
    print('\nRegression p-value:')
    print(res.pvalue)
    
    
    
def plot_policy_slope_CDFs(grp):
    
    figure(figsize = [9,3])
        
    subplot(1,3,1)
    plot_ecdf(grp['hh_lat_b1'])
    title('Latency ')
    xlabel('Slope Estimates\n[ms per trial]')
    ylabel('CDFs\n High-Fast Condition')
    xlim([-10,10])
    yticks([0,0.25,0.5,0.75,1])

    subplot(1,3,2)
    plot_ecdf(grp['hh_dur_b1'])
    title('Duration')
    xlabel('Slope Estimates\n[ms per trial]')
    ylabel('')
    xlim([-100,100])
    yticks([0,0.25,0.5,0.75,1])

    subplot(1,3,3)
    plot_ecdf(grp['hh_hgt_b1'])
    title('Height')
    xlabel('Slope Estimates\n[ms per trial]')
    ylabel('')
    xlim([-0.2,0.2])
    yticks([0,0.25,0.5,0.75,1])

    tight_layout()
    
    
def plot_slopes_corr(grp):
    A = sp.stats.zscore(grp['hh_lat_b1'])
    B = sp.stats.zscore(grp['hh_dur_b1'])

    ids1 = abs(A)<2.5
    ids2 = abs(B)<2.5
    keep = ids1 & ids2

    plt.figure()
    plt.scatter(A[keep], B[keep], c = grp.rew_tot[keep])

    sp.stats.linregress(A[keep], B[keep])

    xlabel('Latency Slope [z-score]')
    ylabel('Duration Slope [z-score]')
    title('Slope Correlation')


def plot_policy_rew_CDFs(grp):
    
    figure(figsize = [3,3])
    plot_ecdf(grp.rew_t0)
    plot_ecdf(grp.rew_t1)
    title('Reward Distributions')
    xlabel('Reward')
    ylabel('CDF')
    legend(['Initial Policies', 'Final Policies'])
    tight_layout()
    
    ks_test = sp.stats.ks_2samp(grp.rew_t0, grp.rew_t1)
    
    print('\nKS Statistic:')
    print(ks_test.statistic)
    
    print('\np-value:')
    print(ks_test.pvalue)    



def plot_durations(grp):
    
    # Get optimal and subject policies
    policy_opt = get_policy_opt()
    avg = get_policies(grp)

    # Number of subjects
    ns = np.shape(grp)[0]
    
    # Color subjects by optimality
    cmap = 'viridis'
    
    # Change units to [s]
    avg = avg*1e-3

    # Figure
    figure(figsize = [4,4])

    # Optimal policy * group averages
    plot([0,1,2,3], policy_opt*1e-3, '--ok')
    plot([0,1,2,3], np.mean(avg,0) , '--or')
    
    # Subject data
    scatter(np.random.normal(0,0.1,ns), avg.iloc[:,0], c = grp.rew_tot, cmap = cmap)
    scatter(np.random.normal(1,0.1,ns), avg.iloc[:,1], c = grp.rew_tot, cmap = cmap)
    scatter(np.random.normal(2,0.1,ns), avg.iloc[:,2], c = grp.rew_tot, cmap = cmap)
    scatter(np.random.normal(3,0.1,ns), avg.iloc[:,3], c = grp.rew_tot, cmap = cmap)

    legend(['Optimal', 'Means'])

    # Adjust figure labels
    conds = ['High-Slow', 'Low-Slow', 'High-Fast', 'Low-Fast']
    ax = gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conds, rotation = 45)
    
    ylabel('Stay Durations [s]')
    
    ylim([0,20])
    #grid('on')
    title('Subject Stay-Durations')
    tight_layout()


def plot_differentiation(grp):
    
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
    
    figure(figsize=[4,4])

    scatter(opts[0] , opts[1] , c = 'k', marker = 'x', zorder = 100)
    scatter(means[0], means[1], c = 'r', marker = 'x', zorder = 101)

    scatter(diffs[:,0], diffs[:,1], c = grp.rew_tot)

    xlabel('Best minus Mid Durations [s]')
    ylabel('Mid minus Worst Durations [s]')
    title('Condition Differences')
    legend(['Optimal', 'Mean'])
    
    xlim([0,10])
    ylim([0, 5])
    tight_layout()
    

def plot_differentiation_middle(subj, avg):
    ns = len(subj)
    total_milk = [subj[i].total_milk.iloc[-1] for i in range(0,ns)]
    
    # Change units
    avg = avg*1e-3
    
    figure(figsize=[4,4])
    scatter(total_milk, abs(avg[:,1] - avg[:,2]))
    xlabel('Total Reward [a.u.]')
    ylabel('|Middle Difference| [s]')
    title('Distinguishing Middle Conditions')
    tight_layout()


# Plot policy heat map
def plot_policy_sweep_heatmap():
    
    # Get rewards for a bunch of policies 
    deltas = np.arange(-2500, 8500, 1e2)    # Baseline changes
    scales = np.arange(0, 2.4, 0.1)         # Multiplicative factor on modulation

    rewards = run_policy_sweep(deltas, scales)
    
    # Rescale units on deltas
    deltas = deltas*1e-3
    
    figure(figsize=[4,4])
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
    title('Policy Differences')
    tight_layout()


def plot_policy_space_subjs(grp):
    
    # Get optimal policy
    policy_opt = get_policy_opt()
    
    # Get subject coordinates
    subj_base, subj_delta, subj_mod = policy_space(policy_opt, grp)

    # Plot them
    plt.scatter(subj_mod, subj_delta*1e-3, zorder=10, c = grp.rew_tot)
    
    
def plot_group_rew_by_subj(subj, save = False, path = './'):
    ns = len(subj)
    total_milk = [subj[i].total_milk.iloc[-1] for i in range(0,ns)]
    
    fig = figure()
    plot(np.arange(0,ns), total_milk,'o')
    title('Performance, All')
    xlabel('Subject Number')
    ylabel('Total Reward')
    grid()
    tight_layout()
    
    if save:
        fig.savefig(path + 'all-performance.png')
        close(fig)

    return fig
    

def plot_group_rew_by_nt(subj, save = False, path = './'):
    ns = len(subj)
    
    n = [len(subj[i].time_to_response) for i in range(0,ns)]
    t = [subj[i].total_milk.iloc[-1] for i in range(0,ns)]
    
    fig = figure()
    fig.set_size_inches(5, 4)
    plot(n, t, 'o')
    grid()
    xlabel('Number of trials')
    ylabel('Total Reward')
    title('Reward by Trial Count')
    
    if save:
        fig.savefig(path + 'all-rew-by-ntrials.png')
        close(fig)

    return fig
    
    
def plot_group_rew_by_avg_latency(subj, save = False, path = './'):

    ns = len(subj)

    m = [np.median(subj[i].time_to_response) for i in range(0,ns)]
    t = [subj[i].total_milk.iloc[-1] for i in range(0,ns)]
    
    fig = figure()
    fig.set_size_inches(5, 4)
    plot(m, t, 'o')
    grid()
    xlabel('Subject Median Latency')
    ylabel('Total Reward')
    title('Reward by Latency')
    
    if save:
        fig.savefig(path + 'all-rew-by-latency.png')
        close(fig)

    return fig


def plot_group_rew_by_mad_latency(subj, save = False, path = './'):

    ns = len(subj)

    s = [mad(subj[i].time_to_response) for i in range(0,ns)]
    t = [subj[i].total_milk.iloc[-1] for i in range(0,ns)]
    
    fig = figure()
    fig.set_size_inches(5, 4)
    plot(s, t, 'o')
    grid()
    xlabel('Subject Latency MAD')
    ylabel('Total Reward')
    title('Reward by Latency Variability (MAD)')
    
    if save:
        fig.savefig(path + 'all-rew-by-latency-mad.png')
        close(fig)

    return fig


def plot_group_rew_by_rbcv(subj, save = False, path = './'):

    ns = len(subj)

    s = [mad(subj[i].time_to_response) for i in range(0,ns)]
    m = [np.median(subj[i].time_to_response) for i in range(0,ns)]
    t = [subj[i].total_milk.iloc[-1] for i in range(0,ns)]
    
    fig = figure()
    fig.set_size_inches(5, 4)
    plot(np.array(s)/np.array(m), t, 'o')
    grid()
    xlabel('Subject Robust Latency CV')
    ylabel('Total Reward')
    title('Reward by Latency CV')
    
    if save:
        fig.savefig(path + 'all-rew-by-latency-rcv.png')
        close(fig)

    return fig

def plot_group_rew_by_num_outliers(subj, save = False, path = './'):

    ns = len(subj)

    c = [sum( subj[i].time_to_response - np.median(subj[i].time_to_response) > 3*np.std(subj[i].time_to_response)) for i in range(0,ns)]
    t = [subj[i].total_milk.iloc[-1] for i in range(0,ns)]

    fig = figure()
    fig.set_size_inches(5, 4)
    plot(np.array(c) + np.random.normal(0,0.1,size=[ns]), t, 'o')
    grid()
    xlabel('Jittered Subject Outlier Count')
    ylabel('Total Reward')
    title('Reward by Outlier Count')
    
    if save:
        fig.savefig(path + 'all-rew-by-num-outliers.png')
        close(fig)

    return fig


def plot_optimal_durations(times, best_ind):
    fig = figure()
    fig.set_size_inches(5, 4)
    plot(np.arange(0,4), np.transpose(times[best_ind,:]),'o')
    grid('on')
    xlabel('Patch Type')
    ylabel('Duration [ms]')
    title('Optimal Switch-times')
    fig.savefig('./optimal-switch-times.png')


# These don't comply with the group call signature so need to be called separately
# (and also need to not be called plot_group_*)
def plot_rew_by_err_group(e,t):
    fig = figure()
    fig.set_size_inches(5, 4)
    plot(e, t, 'o')
    grid('on')
    xlabel('Response Profile RMSE [ms]')
    ylabel('Total Reward')
    title('Response Optimality and Reward')
    fig.savefig('./all-rew-by-resp-error.png')
        
def plot_dur_err_change_group(err):
    plot(err[:,0], err[:,1], 'o')
    match_lims()


##


# Plot improvement
# plot_group_dur_err_change(err)

## ---------------------------------------------- Misc plots ----------------------------------------------- ##
def plot_():
    figure()
    plot(aes,nt,'o')
    xlabel('AES Score')
    ylabel('Number of Trials in Exp')
    title('Apathy and Trial Counts')
    tight_layout()
    res = sp.stats.linregress(aes, nt)

    fig = figure()
    fig.set_size_inches([4.07, 3.55])
    plot(aes, latency,'o')
    xlabel('AES Score')
    ylabel('Latency')
    title('Apathy and Latency')
    tight_layout()
    res = sp.stats.linregress(aes, latency)
    
    fig = figure()
    fig.set_size_inches([4.07, 3.55])
    plot(aes, duration,'o')
    xlabel('AES Score')
    ylabel('Duration')
    title('Apathy and Duration')
    tight_layout()
    res = sp.stats.linregress(aes, duration)
    
    fig = figure()
    
    fig.set_size_inches(11.5,  2.25)
    
    subplot(1,4,1)
    plot(aes, avg[:,0],'o')
    xlabel('AES Score')
    ylabel('Stay Duration')
    title('Trial Type HH')
    res = sp.stats.linregress(aes, avg[:,0])
    
    subplot(1,4,2)
    plot(aes, avg[:,1],'o')
    xlabel('AES Score')
    title('Trial Type HL')
    res = sp.stats.linregress(aes, avg[:,1])
    
    subplot(1,4,3)
    plot(aes, avg[:,2],'o')
    xlabel('AES Score')
    title('Trial Type LH')
    res = sp.stats.linregress(aes, avg[:,2])
    
    subplot(1,4,4)
    plot(aes, avg[:,3],'o')
    xlabel('AES Score')
    title('Trial Type LL')
    res = sp.stats.linregress(aes, avg[:,3])
    

    d,v = np.linalg.eig(np.cov(avgs.T))
    fig = figure()
    plot(d/sum(d), '-o')
    ylim([0,1])
    grid('on')
    ylabel('Fraction Variance Explained')
    xlabel('Principal Component')
    title('PCA Variance Explained')
    fig.savefig('./response-var-explained.png')





    
def plot_decomp(policy_opt, lms):
    
    
    pass
    

def save_wrapper(fig, save, path, tstr, snum):
    if save:
        fig.savefig(path + tstr + str(snum).zfill(2) + '.png')
        close(fig)    


def label(xlab = '', ylab = '', tstr = ''):
    xlabel(xlab)
    ylabel(ylab)
    title(tstr)
    tight_layout()
    

def match_lims(add_diag = True):
    grid()
    
    xlims = xlim()
    ylims = ylim()
    
    lb = min(xlims[0], ylims[0])
    ub = max(xlims[1], ylims[1])
    
    xlim([lb,ub])
    ylim([lb,ub])
    
    if add_diag:
        plot([lb,ub], [lb,ub], '--k')
        
        
def plot_ecdf(data):
    
    ns  = len(data)
    cdf = np.cumsum(np.ones(ns))/ns
    
    plot(np.sort(data), cdf)
    
    ylim([0,1])
    yticks( np.arange(0, 1.1, 0.1) )
    grid()
    
    #xlabel('Stay-Duration Change')
    #ylabel('CDF')
    tight_layout()
    
    
def prop(data):
    # Dichotomized at zero
    ns = len(data)
    c1 = sum(data < 0) / ns
    #c2 = sum(data > 0) / ns
    
    #plt.bar([0,1], [c1, c2])
    return c1
    
def jse(data, fn):
    
    nreps = len(data)
    evals = np.zeros(nreps)
    fval  = fn(data)
    
    inds = np.arange(nreps)
    for i in range(0, nreps):
        sub = inds[inds !=i ]
        evals[i] = fn( data[sub] )
    
    se = np.sqrt( (nreps - 1)/nreps * sum( (evals - fval)**2 ) )
    
    return se
