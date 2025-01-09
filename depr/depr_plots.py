from plot_foraging import *

def plot_policy_models(subj, grp, show_title = False):

    # Get policy models and associated info
    model_ve, model_rve, model_beta, model_proj = get_policy_models(subj, grp)

    # Plot Reduced policy model scores
    plot_policy_sweep_heatmap()     # Policy space: reward contours in duration vs differntiation
    plot_policy_space_subjs(grp)    # Subjects' policies in policy space

    figure(figsize = [3.3,3])
    plt.bar(range(0,3), model_ve[1:4])    
    plt.bar(range(3,6), model_rve[1:4])

    plt.xticks(range(0,6), labels = ['PC1', 'Shift\nand\nScale', 'Shift\nOnly', 'PC1', 'Shift\nand\nScale', 'Shift\nOnly'])#, rotation = 30, ha='right', rotation_mode='anchor')

    #xlabel('Models', rotation = 30)
    legend(['Policy','Reward'])
    ylabel('Fraction Variance Explained')
    
    if show_title:
        title('Model Comparison')
    
    tight_layout()




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



def plot_differences(grp):

    durs = np.array(grp[ ['hl_dur_avg', 'll_dur_avg', 'hh_dur_avg', 'lh_dur_avg'] ])
    evals, evecs = np.linalg.eig(np.cov(durs.T))
    scores = (durs - np.mean(durs,axis=0))@ evecs[:,0]/10000

    plt.figure(figsize=(16,4))
    plt.subplot(1,4,1)
    #plt.plot(np.mean(durs[:,   :],axis=0), '-o', label='mean, all')
    plt.plot(np.mean(durs[:234,:],axis=0), '-o', label='mean, young')
    plt.plot(np.mean(durs[234:,:],axis=0), '-o', label='mean, old')
    plt.legend()
    plt.xlabel('Condition')
    plt.xticks(np.arange(4), ['hl', 'll', 'hh', 'lh'])
    plt.ylabel('Duration (ms)')


    plt.subplot(1,4,2)
    evals, evecs_all   = np.linalg.eig(np.cov(durs.T))
    evals, evecs_young = np.linalg.eig(np.cov(durs[:234,:].T))
    evals, evecs_old   = np.linalg.eig(np.cov(durs[234:,:].T))

    #plt.plot(evecs_all[:,0],'-o', label='PC1 (all)')
    plt.plot(-evecs_young[:,0],'-o', label='PC1 (young)')
    plt.plot(evecs_old[:,0],'-o', label='PC1 (old)')
    plt.legend()
    plt.xlabel('Condition')
    plt.xticks(np.arange(4), ['hl', 'll', 'hh', 'lh'])
    plt.ylabel('Weight')

    plt.subplot(1,4,3)
    plt.plot(np.mean(durs,axis=0)/np.linalg.norm(np.mean(durs,axis=0)), '-o', label='mean')

    diff = np.mean(durs[235:,:],axis=0) - np.mean(durs[:235,:],axis=0)
    diff = diff / np.linalg.norm(diff)

    plt.plot(evecs[:,0],'-o', label='PC1 (all)')
    plt.plot(diff, '-o', label='Age diff.')
    plt.legend()

    plt.xlabel('Condition')
    plt.xticks(np.arange(4), ['hl', 'll', 'hh', 'lh'])
    plt.ylabel('Normalized Vector Component')

    plt.subplot(1,4,4)
    scores = (durs - np.mean(durs,axis=0))@ evecs[:,0]/10000

    support = np.arange(-10000, 25000, 100)/10000
    kde1 = sp.stats.gaussian_kde(scores[:234])
    kde2 = sp.stats.gaussian_kde(scores[234:])
    plt.plot(support,kde1(support), label='young')
    plt.plot(support,kde2(support), label='old')
    plt.legend()
    plt.xlabel('PC1 Scores [10 s]')
    plt.ylabel('Density Estimate')

    plt.tight_layout()


def plot_rt_percentiles():

    rts = grp.rt_med.values
    rts = rts[~np.isnan(rts)]

    support = np.arange(0, np.max(rts), 100)
    kde1 = sp.stats.gaussian_kde(rts)

    cdf = np.cumsum(kde1(support))
    cdf = cdf / np.max(cdf)

    plt.figure(figsize=[6,3])
    plt.subplot(1,2,1)
    plt.plot(support, kde1(support))
    plt.grid()
    plt.title('RT Median PDF')

    plt.subplot(1,2,2)
    plt.plot(support, cdf)
    plt.grid()
    plt.title('RT Median CDF')

    grp['rt_pass'] = grp.rt_med < 1000

    plt.tight_layout()

def plot_rt_by_scores(grp, scores):
    durs = np.array(grp[ ['hl_dur_avg', 'll_dur_avg', 'hh_dur_avg', 'lh_dur_avg'] ])
    med_dur = np.median(durs, axis = 1)

    plt.figure(figsize = [6,3])
    plt.subplot(1,2,1)
    plt.plot(grp.rt_med[grp.rt_pass], scores[grp.rt_pass],'o')
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(grp.rt_med[grp.rt_pass], med_dur[grp.rt_pass],'o')
    plt.grid()



    
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



def plot_stay_lms_with_performance(subj, grp, model='marginal'):

    if model == 'marginal':
        results = fit_stay_lms_marginal(subj)
    elif model == 'full':
        results = fit_stay_lms_full(subj)
    
    inds = np.argsort(results['r2']).values
    xs = np.arange(len(inds))

    # Plot R values
    plt.figure(figsize=[3.3,3])
    plt.scatter(xs, results['r2'].values[inds], zorder=10, c=grp.rew_tot, s=5)
    plt.ylim([0,1])
    plt.grid()
    plt.xlabel('Subject Rank')
    plt.ylabel('R-squared')
    plt.tight_layout()

    # Plot log of model F-statistic p-values
    plt.figure(figsize=[3.3,3])
    plt.scatter(xs, np.log(results['fp'].values[inds]), zorder=10, c=grp.rew_tot, s=5)
    plt.grid()
    plt.xlabel('Subject Rank')
    plt.ylabel('log(F-statistic p-value)')
    plt.tight_layout()

    # Plot model start coefficients
    plt.figure(figsize=[3.3,3])
    plt.scatter(xs, results['b_h-'].values[inds], zorder=10, c=grp.rew_tot, s=5)
    plt.grid()
    plt.ylim([0,6000])
    plt.xlabel('Subject Rank')
    plt.ylabel('Initial Reward Rate Coefficient')
    plt.tight_layout()

    # Plot model 
    plt.figure(figsize=[3.3,3])
    plt.scatter(xs, results['b_-h'].values[inds], zorder=10, c=grp.rew_tot, s=5)
    plt.grid()
    plt.ylim([-7000,0])
    plt.xlabel('Subject Rank')
    plt.ylabel('Reward Decay Rate Coefficient')
    plt.tight_layout()

    # plt.figure(figsize=[3.3,3])
    # plt.grid()
    # plt.xlabel('Subject Rank')
    # plt.ylabel('High-Low Coefficient')
    # plt.tight_layout()


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
    plt.plot(support, kde1(support), label='Young')
    plt.plot(support, kde2(support), label='Old')
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

