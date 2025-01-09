import numpy as np

def read_subjects(max_ns = 1000, spath = './data/round-2/'):
    
    ##
    ## 1st directory structure from Noham
    ##
    # Import subjects, sorting files so indexing is consistent between reads
    # files = [join(spath, f) for f in sorted(listdir(spath)) if isfile(join(spath, f))]
    
    ##
    ## 2nd directory structure from Noham
    ##
    # Get file names for milkman task data
    #dirs  =  [join(spath, d) for d in sorted(listdir('./data/controls')) if d[0:4] == 'Milk']

    # Get foraging task files
    #fname_foraging = 'MilkManOnline.csv'
    #files_foraging = [join(d, fname_foraging) for d in dirs]

    # Get AES files
    #fname_aes = 'AES_with_attention_questions.csv'
    #files_aes = [join(d, fname_aes) for d in dirs]

    ##
    ## 3rd directory structure from Noham
    ##
    files_foraging = [join(spath, f) for f in sorted(listdir(spath)) if f[0:4] == 'Milk']
    files_aes      = [join(spath, f) for f in sorted(listdir(spath)) if f[0:3] == 'AES' ]

    # Put all files in a dataframe
    files = pd.DataFrame([files_foraging, files_aes]).T

    # Number of subjects
    ns = min(len(files_foraging), max_ns)
    
    # Compile list of subject data frames and list of associated masks
    #subj = pd.DataFrame([])
    subj = []
    grp  = []
    #files_kept = files_foraging[0:ns]
    

    # Read all subject data:
    # Other notes:
    # Subject 6 appears to have more reward than theoretically possible...
    #
    # Exclusions:
    # Subjects  5, 13, 35 fail attention check & 35 has bad foraging data
    # Subjects 18, 27, 41 appear to have bad rew curves
    # Subjects 44, 48     are big duration and latency outliers, respectively
    # skip = [6, 5,13,35, 18,27,41, 44,48]
    skip = [45, 89, 99]
    reindex = 0
    for i in range(0, ns):
        # Skip bad subjects
        if i in skip:
            continue
        
        # Read
        sdata       = read_subj_foraging_file(files_foraging[i])
        score, attn = read_subj_aes_file(files_aes[i])
        
        # Some subjects have empty data.
        if len(sdata) > 0:
                            
            # Pandas dataframe format
            # snum = np.ones([nrows], dtype = 'int')*reindex
            # sdata['snum'] = snum
            # subj = pd.concat([subj, sdata])
            
            # List-of-subjects data format
            subj.append(sdata)
            
            # Save apathy data
            grp.append(score)
            
            # Increment subject counter
            reindex += 1
        else:
            print('Subject ' + str(i) + ' has data length 0.')
            #files_kept = listrm(files_kept, i)
    
        if not attn:
            print('Subject ' + str(i) + ' failed attention check.')
                
    # Recompute number of subjects
    ns = ns - len(skip)

    # Convert group data to dataframe
    grp = pd.DataFrame(grp, columns = ['aes'])

    return subj, grp, ns, files


# OLD
def task_analysis():

    # Task parameters
    switch_cost = 4     # Switch cost in [s]
    exp_time    = 600   # Experiment length in [s]
    
    # Conversion
    ms_per_s = 1000
    
    base  = [0.02  , 0.01  ]
    decay = [0.0004, 0.0002]

    # Possible reward rates, from very low to the max for low start-rate patches
    rates = np.arange(0.0001, 0.0075, 0.0001)

    rew = 0
    k = 0
    policy = np.zeros([len(rates),4])
    total_time = switch_cost * ms_per_s
   
    for i in range(0,2):
        for j in range(0,2):
            policy[:,k] = get_time_from_rew_rate(rates, decay[j], base[i])
            
            total_time += policy[:,k] * (1/4) 
            rew += 1/4 * get_integrated_rew(base[i], decay[j], policy[:,k])
            k += 1

    num_trials = get_trial_count(exp_time, switch_cost, policy)

    
    # Reward rate for experiment  
    rew_rate = rew / total_time * ms_per_s
    
    # Total reward over experiment
    total_rew = rew_rate * exp_time
    
    # Adjustment for non-decision time
    total_rew_adjusted = rew_rate * (exp_time - num_trials * 0.5) 

    # Best possible total reward
    best_ind = [i for i, x in enumerate(total_rew_adjusted == max(total_rew_adjusted)) if x]

    return total_rew, total_rew_adjusted, policy, best_ind

def get_time_from_rew_rate(rew_rate, decay, base):
    return (np.log(base) - np.log(rew_rate)) / decay

def get_integrated_rew(base, decay, time):
    return 1/100 *base / decay * (1 - np.exp(-decay*time))

def get_stay_dur(rew_rate, decay, base):
    return - 1/decay * np.log(rew_rate)/np.log(base)

def get_trial_count(exp_time, switch_cost, t):
    ms_per_s = 1000
    return (exp_time + switch_cost)/(switch_cost + 1/4 * np.sum(t,1)/ms_per_s) 

#def R(rew_rate, decay, base, T, m):
#    N_ = N(rew_rate, decay, base, T,m)
#    t2 = a*np.sum(1 - 1/decay)
#    return N_*t2


def plot_trial_rels(df, msk, snum, save, path):
    fig = figure()
    fig.set_size_inches(5, 4)
    plot(df.space_down_time[ msk.hh], df.height[msk.hh], 'o')
    plot(df.space_down_time[ msk.hl], df.height[msk.hl], 'o')
    plot(df.space_down_time[ msk.lh], df.height[msk.lh], 'o')
    plot(df.space_down_time[ msk.ll], df.height[msk.ll], 'o')
    title('Observed Relations, Subject ' + str(snum))
    xlabel('Hold Time')
    ylabel('Height')
    ylim([0,100])
    xlim([0,40000])
    grid()
    tight_layout()
    
    if save:
        fig.savefig(path + 'height-by-duration-' + str(snum).zfill(2) + '.png')
        close(fig)

    return fig


def plot_trial_latency_avgs(df, msk, snum, save, path):
    fig = figure()
    fig.set_size_inches(5, 4)
    
    ttr_avg_hh, ttr_std_hh = stats(df.time_to_response[msk.hh])
    ttr_avg_hl, ttr_std_hl = stats(df.time_to_response[msk.hl])
    ttr_avg_lh, ttr_std_lh = stats(df.time_to_response[msk.lh])
    ttr_avg_ll, ttr_std_ll = stats(df.time_to_response[msk.ll])
    
    avgs = [ttr_avg_hh, ttr_avg_hl, ttr_avg_lh, ttr_avg_ll]
    stds = np.array([ttr_std_hh, ttr_std_hl, ttr_std_lh, ttr_std_ll])
    
    errorbar(np.arange(0,4), avgs, 2*stds, fmt='o')
    
    xlabel('Patch Type')
    ylabel('Latency [ms]')
    title('Latency by Type, Subject ' + str(snum))
    ylim([0,2000])
    grid()
    tight_layout()
    
    if save:
        fig.savefig(path + 'latency-by-patch-' + str(snum).zfill(2) + '.png')
        close(fig)

    return fig




def plot_trial_durations(df, msk, snum, save, path):
    fig = figure()
    fig.set_size_inches(5, 4)
    
    ttr_avg_hh, ttr_std_hh = stats(df.space_down_time[msk.hh])
    ttr_avg_hl, ttr_std_hl = stats(df.space_down_time[msk.hl])
    ttr_avg_lh, ttr_std_lh = stats(df.space_down_time[msk.lh])
    ttr_avg_ll, ttr_std_ll = stats(df.space_down_time[msk.ll])
    
    avgs = [ttr_avg_hh, ttr_avg_hl, ttr_avg_lh, ttr_avg_ll]
    stds = np.array([ttr_std_hh, ttr_std_hl, ttr_std_lh, ttr_std_ll])
    
    errorbar(np.arange(0,4), avgs, 2*stds, fmt='o')
    
    xlabel('Patch Type')
    ylabel('Duration [ms]')
    title('Stay Duration by Type, Subject ' + str(snum))
    ylim([0,20000])
    grid()
    tight_layout()
    
    if save:
        fig.savefig(path + 'duration-by-patch-' + str(snum).zfill(2) + '.png')
        close(fig)

    return fig









def plot_condition_dprimes(subj, avg, policy_opt):
    
    ns = len(subj)
    total_milk = [subj[i].total_milk.iloc[-1] for i in range(0,ns)]
    
    # Color subjects by optimality
    cmap = 'viridis'
    
    # Change units to [s]
    avg = avg*1e-3

    # Figure
    figure(figsize = [4,4])

    # Means and medians
    means = np.mean(avg,0)
    stds  = np.std( avg,0)

    # Effect sizes
    d = np.zeros([4,4])
    for i in range(0,4):
        for j in range(i+1,4):
            d[i,j] = 2*(means[i] - means[j])/(stds[i] + stds[j])

    # Matrix of effect sizes
    imshow(d, interpolation = 'None')



import numpy as np



def get_subj_durations(df, msk):
    
    ttr_avg_hh, ttr_std_hh = stats(df.space_down_time[msk.hh])
    ttr_avg_hl, ttr_std_hl = stats(df.space_down_time[msk.hl])
    ttr_avg_lh, ttr_std_lh = stats(df.space_down_time[msk.lh])
    ttr_avg_ll, ttr_std_ll = stats(df.space_down_time[msk.ll])
    
    avgs = [ttr_avg_hh, ttr_avg_hl, ttr_avg_lh, ttr_avg_ll]
    stds = np.array([ttr_std_hh, ttr_std_hl, ttr_std_lh, ttr_std_ll])
    
    return avgs, stds
    


avgs = np.zeros([4,ns])
stds = np.zeros([4,ns])
e = np.zeros([ns])
for i in range(0,ns):
    avgs[:,i], stds[:,i] = get_subj_durations(subj[i], msks[i])
    d = (np.squeeze(times[best_ind,:])-avgs[:,i])
    e[i] = np.sqrt(sum(d*d))

t = [subj[i].total_milk.iloc[-1] for i in range(0,ns)]


rew = 0
total_time = switch_time
times = np.arange(2000, 20000, 100)
times_mat = np.zeros([len(times),4])
times_mat[:,0] = times
times_mat[:,1] = times
times_mat[:,2] = times
times_mat[:,3] = times
k = 0
for i in range(0,2):
    for j in range(0,2):
        rew += 1/4 * patch_rew(s[i], l[j], times_mat[:,k])
        total_time += times_mat[:,k] * (1/4) 
        k += 1

nt_uni = N(600000, 4000, times_mat)

rew_rate = rew / total_time * ms_per_s
total_rew_uni = rew_rate * exp_time
total_rew_uni_adjusted = rew_rate * (exp_time - nt_uni * 0.5) 



fig = plot_group_rew_by_nt(subj)
plot(nt, total_rew)
plot(nt, total_rew_adjusted)
plot(nt_uni, total_rew_uni_adjusted)
fig.legend(['Subjects', 'Fixed Rate Slice', 'Adjusted F.R.S.', 'Adjusted Fixed Time Slice'], loc = 'lower right')
fig.savefig('./reward-by-trial-vs-theory.png')


d,v = np.linalg.eig(np.cov(avgs))
fig = figure()
plot(d/sum(d), '-o')
ylim([0,1])
grid('on')
ylabel('Fraction Variance Explained')
xlabel('Principal Component')
title('PCA Variance Explained')
fig.savefig('./response-var-explained.png')

fig = figure()
plot(v[:,0], '-o')
plot(v[:,1], '-o')
grid('on')
xlabel('Patch Type')
ylabel('Duration Loading')
title('Top 2 Response Variation PCs')
fig.savefig('./top-response-pcs')



d,v = np.linalg.eig(np.cov(avgs))
fig = figure()
plot(d/sum(d), '-o')
ylim([0,1])
grid('on')
ylabel('Fraction Variance Explained')
xlabel('Principal Component')
title('PCA Variance Explained')
fig.savefig('./response-var-explained.png')

fig = figure()
plot(v[:,0], '-o')
plot(v[:,1], '-o')
grid('on')
xlabel('Patch Type')
ylabel('Duration Loading')
title('Top 2 Response Variation PCs')
fig.savefig('./top-response-pcs')
