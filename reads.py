import pandas as pd
import numpy  as np

def read_subjects(spath = './subj_data/', summary_only = False):
    # Number of subjects, subject data list to put foraging frames in
    ns, subj = 351, []

    # Read all subject files
    for snum in range(1, ns):

        # Subject foraging file
        subj.append( read_subject_file_for(snum = snum, spath = spath) )

        # Get data from additional mesures
        aes_df = read_subject_file_survey(snum = snum, spath = spath, summary_only = summary_only, survey = 'aes')
        phq_df = read_subject_file_survey(snum = snum, spath = spath, summary_only = summary_only, survey = 'phq')
        tol_df = read_subject_file_tol(snum = snum, spath = spath)
        rtt_df = read_subject_file_rtt(snum = snum, spath = spath)
        
        # Merge the summary data
        joined = pd.concat([aes_df, phq_df, tol_df, rtt_df], axis=1)

        # If this is the first read, initialize columns for group data
        if snum == 1:
            # Initialize group data
            grp = pd.DataFrame(index=range(0,ns-1), columns = joined.columns.tolist())
        
        # Save merged data
        grp.iloc[snum-1] = joined

    # Get ages
    age = pd.read_csv(spath + 'age.csv').sort_values('SubIndex').reset_index(drop=True)
    grp['age'] = age['Age']

    # Categorical age variable
    grp['age_cat'] = 0
    grp.loc[grp['age'] > 50, 'age_cat'] = 1

    # Convert main columns to floats (some will be "object" due to earlier missing data)
    for col in grp.columns:
        if 'qc' not in col:
            grp[col] = grp[col].astype(float)
        else:
            grp[col] = grp[col].astype(bool)

    return subj, grp

def drop_subjects(subj, grp, droplist, reset_index = True):
    # Make sure droplist is sorted and has only unique values
    droplist = np.sort(np.unique(droplist))

    # Remove subjects from subject list and group data
    if reset_index:
        subj = [s for snum, s in enumerate(subj) if snum not in droplist]
        grp = grp.drop(droplist).reset_index(drop = True)
    else:
        for snum, s in enumerate(subj):
            if snum in droplist:
                subj[snum][:] = np.nan
        grp = grp.drop(droplist)

    return subj, grp


def read_subject_file_for(snum, spath):
    # Construct filena,e
    file = spath + 'subject_' + str(snum) + '_milkman.csv'
    
    # Load participant data
    df = pd.read_csv(file)
    
    # Fix annoying column names
    df = fix_column_names(df) 
    
    # Get test trials
    msk = df.trial_type.isin(['t','test'])
    
    # Subset data to remove practice info
    df = df.loc[msk,:]
    
    # Get the trial types
    conds = pd.DataFrame()

    # Some names are different in new data
    if 'decay' in df.columns:
        df = df.rename(columns = {'decay':'decay_rate'})
        df = df.rename(columns = {'milk_earned':'height'})

    # Conditions (ordered here from best to worst)
    conds['hl'] = (df.S == 0.02) & (df.decay_rate == 0.0002)
    conds['ll'] = (df.S == 0.01) & (df.decay_rate == 0.0002)
    conds['hh'] = (df.S == 0.02) & (df.decay_rate == 0.0004)
    conds['lh'] = (df.S == 0.01) & (df.decay_rate == 0.0004)
    
    # Dummy condition for selecting everything
    conds['ac'] = True

    # Columns to keep
    #keep = ['trial_number', 'trial_type', 'S',
    #       'decay_rate', 'time_to_response', 'space_down_time', 'height',
    #       'round_milk', 'total_milk', 'milk_reward_scale', 'travel_time_param']
    keep = ['trial_number', 'S', 'decay_rate', 'time_to_response',
            'space_down_time', 'height', 'total_milk']

    df = df[keep]
    df = pd.concat([df,conds], axis = 1)

    return df


def read_subject_file_survey(snum, spath, summary_only = True, survey = 'aes'):
    # Construct filename, load participant data, fix column names
    file = spath + 'subject_' + str(snum) + '_'+ survey + '.csv'
    df = fix_column_names( pd.read_csv(file) )

    # Rename columns on new data for consistency
    if 'is_attention' in df.columns:
        df = df.rename(columns = {'is_attention':'attention', 'score':'answer_score'})
    
    # Attention check, survey score, and RT statistics
    pass_attn_check = sum(df.answer_score[df.attention == 1]) == sum(df.attention)    
    score = sum(df.answer_score[df.attention == 0])
    median, mad, lapse = get_loc_scale_lapse(df['RT'])

    # Save to output
    summaries = {survey+'_score':score, survey+'_rt_med':median, survey+'_rt_mad':mad, survey+'_rt_lapse':lapse, survey+'_qc':pass_attn_check}
    
    # Specific question answers:
    if summary_only is False:
        answers = {(survey+'_'+str(i)):value for i,value in enumerate(df.loc[df.attention == 0,:]['answer_score'].values)}
        summaries.update(answers)

    # Output as dataframe
    return pd.DataFrame(summaries, index = [0])


def read_subject_file_tol(snum, spath):
    # Construct filename
    file = spath + 'subject_' + str(snum) + '_tol.csv'

    # Load participant data
    try:
        df = pd.read_csv(file)
    except:
        print('Failed to read file: ' + file)
        df = pd.DataFrame()

    # Check if data is empty
    if len(df) != 0:
        accuracy = sum(df.is_correct)

        rtcs = df['RT'][df.is_correct == 1]
        rtc_med, rtc_mad, rtc_lapse = get_loc_scale_lapse(rtcs)

        rtis = df['RT'][df.is_correct == 0]
        rti_med, rti_mad, rti_lapse = get_loc_scale_lapse(rtis)
    else:
        print('Data length zero in file: ' + file)
        accuracy, rtc_med, rtc_mad, rtc_lapse, rti_med, rti_mad, rti_lapse = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    out = pd.DataFrame({'tol_acc':accuracy, 'tol_rtc_med':rtc_med, 'tol_rtc_mad':rtc_mad, 'tol_rtc_lapse':rtc_lapse,
                        'tol_rti_med':rti_med, 'tol_rti_mad':rti_mad, 'tol_rti_lapse':rti_lapse}, index = [0])

    return out

def read_subject_file_rtt(snum, spath):
    # Construct filename
    file = spath + 'subject_' + str(snum) + '_rt.csv'
    
    # Attempt to load participant data
    try:
        df = pd.read_csv(file)

        # Get RTs for correct trials only
        rts = df['RT'].loc[df['is_correct']==1]

        # Compute RT statistics
        median, mad, lapse = get_loc_scale_lapse(rts)
    except:
        print('Failed to read file: ' + file)
        median, mad, lapse = np.nan, np.nan, np.nan

    # Package and return
    out = pd.DataFrame({'rtt_rt_med':median, 'rtt_rt_mad':mad, 'rtt_rt_lapse':lapse}, index = [0])
    return out

def fix_column_names(df):
    df.columns = df.columns.str.replace(' ', '_'   , regex = True)
    df.columns = df.columns.str.replace('\[%\]', '', regex = True)
    return df

def get_loc_scale_lapse(vec):
    # Return NaNs if no data
    if len(vec) == 0:
        return np.nan, np.nan, np.nan
    #mean   = np.mean(vec)
    #std    = np.std(vec)
    median = np.median(vec)
    mad    = np.median(np.abs(vec - np.median(vec)))
    lapse  = sum(vec > median + 2*1.25*mad)/len(vec)
    #stats  = {'mean':mean, 'std':std, 'median':median, 'mad':mad, 'lapse':lapse}
    return median, mad, lapse
