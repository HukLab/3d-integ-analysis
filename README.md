## Folder structure

* __bin/ (.py)__ - data analysis scripts

* __data/ (.csv)__ - original data collected from two different experiments

* __notes/ (.md, .txt)__ - miscellaneous scratch notes

## Data analysis scripts (in bin/)

### I/O
    trial.py - used by mat_io.py
    mat_io.py - matlab data to csv
    pd_io.py - loading data in pandas
    settings.py - specifies valid conditions in data
    fig_data_gen.sh - batch creation of all data files

### Fitting
    sample.py - bootstrapping
    mle.py - generic fitting helpers
    saturating_exponential.py - sat-exp function
    fit_pmf.py - psychometric functions fitting
    fit_elbows.py - used by fit_pmf.py
    fit_pcor_vs_dur.py - sat-exp fitting
    fit_elbows_alts.py - alternative fits to sensitivity vs duration

### Calculating statistics
    elb_ci.py - confidence intervals and histograms of elbow fit params
    elb_bic.py - calculate BIC for elbow fits of sensitivity vs duration
    summaries.py - overview of # sessions and trials per subject and condition
    sequential_effects.py - comparing inter-trial dependencies
    significance.py - comparing two data collection conditions
    corr.py - for assessing validity of threshold estimates

### Utils for local plotting
    scatter.py - plotting individual subject fit params
    plot_pmf.py - plotting pmfs
    plot_elbs.py - plotting bootstrapped fits to sensitivity vs duration
    plot_resid.py - residuals of log-log plot using different numbers of elbows    
    pcor_mesh_plot.py - plotting accuracy vs. both coh and dur