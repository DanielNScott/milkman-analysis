


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
