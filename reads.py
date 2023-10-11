#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:42:45 2023

@author: dan
"""
import pandas as pd
import numpy  as np

from os import listdir
from os.path import join
#from os.path import isfile

def read_subjects(max_ns = 1000):
    
    # Top directory for subject files
    spath = './data/controls/'
    
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
        if not i in skip:
            sdata       = read_subj_foraging_file(files_foraging[i])
            score, attn = read_subj_aes_file(files_aes[i])
            
            # Confirm all subject files are matched (non-rep paranoia)
            # sidf = files_foraging[i][-28:]
            # sida =      files_aes[i][-28:]
            
            # print('\nChecking subject files are matched')
            # print(sidf == sida)
            # print(sidf)
            # print(sida)
            
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


def read_subj_foraging_file(file):
    # Skip rows
    skiprows = []
    
    # Load participant data
    df = pd.read_csv(file, skiprows = skiprows)
    
    # Fix annoying column names
    df = fix_column_names(df) 
    
    # Get test trials
    msk = df.trial_type == 'test'
    
    # Subset data to remove practice info
    df = df.loc[msk,:]
    
    # Get the trial types
    conds = pd.DataFrame()

    # Conditions (ordered here from best to worst)
    conds['hl'] = (df.S == 0.02) & (df.decay_rate == 0.0002)
    conds['ll'] = (df.S == 0.01) & (df.decay_rate == 0.0002)
    conds['hh'] = (df.S == 0.02) & (df.decay_rate == 0.0004)
    conds['lh'] = (df.S == 0.01) & (df.decay_rate == 0.0004)
    
    # Dummy condition for selecting everything
    conds['ac'] = True
    
    # Columns to keep
    keep = ['trial_number', 'trial_type', 'S',
           'decay_rate', 'time_to_response', 'space_down_time', 'height',
           'round_milk', 'total_milk', 'milk_reward_scale', 'travel_time_param']

    df = df[keep]
    df = pd.concat([df,conds], axis = 1)

    return df


def read_subj_aes_file(file, attn_check = False):
    # 
    skiprows = []
    
    # Load participant data
    df = pd.read_csv(file, skiprows = skiprows)
    
    # Fix annoying column names
    df = fix_column_names(df) 
    
    # Check that subject passes the attention questions
    if attn_check:
        pass_attn_check = sum(df.answer_score[df.attention]) == 2
        
        # Get normal question score
        score = sum(df.answer_score[~df.attention])

    else:
        pass_attn_check = True
    
        # No attention subsetting
        score = sum(df.answer_score)

    
    return score, pass_attn_check


def fix_column_names(df):
    df.columns = df.columns.str.replace(' ', '_'   , regex = True)
    df.columns = df.columns.str.replace('\[%\]', '', regex = True)
    return df
    