## I/O
    trial.py - used by mat_io.py
    mat_io.py - matlab data to csv
    pd_io.py - loading data in pandas
    settings.py - specifies valid conditions in data
    fig_data_gen.sh - batch creation of all data files

## Fitting
    sample.py - bootstrapping
    mle.py - generic fitting helpers
    saturating_exponential.py - sat-exp function
    pmf_fit.py - psychometric functions fitting
    pmf_elbows.py - used by pmf_fit.py
    pcorVsDurByCoh.py - sat-exp fitting

## Plotting
    elb_ci.py - confidence intervals and histograms of elbow fit params
    plotResid.py - residuals of log-log plot using different numbers of elbows    

## Utils for data sanity-checking
    summaries.py - overview of # sessions and trials per subject and condition
    sequential_effects.py - comparing inter-trial dependencies
    significance.py - comparing two data collection conditions
    corr.py - for assessing validity of threshold estimates

## Utils for local plotting
    scatter.py - plotting individual subject fit params
    pmf_plot.py - plotting pmfs
    pmf_plot_bs.py - plotting bootstrapped pmfs
    pcor_mesh_plot.py - plotting accuracy vs. both coh and dur
