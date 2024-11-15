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
