4/12/14
--------
* to fit a _saturating exponential_, there are three parameters: start value, end value (asymptote), and slope
	* if you have fixed start and end value, you only have to find slope!
	* using rmse as cost function, minimize the error between your sample data and the fit for a given slope parameter
		* (i.e. search through range of putative slopes and choose the one yielding a minimum rmse)
* _bootstrapping_
	* given sample X of size N drawn from some population, calculate sample statistic m_X.
	* in general, you can't assume to know that your error in calculating m_X is normally distributed.
	* but, you can resample _from_ X, with replacement, a sample also of size N.
	* if you resample M times, each time calculating your statistic on the resample, this is a bootstrap
	* the _bootstrap's_ standard error now _is_ gaussian
		* S.E. (bootstrap statistic) = rmse of difference between each bootstrap statistic and original sample statistic
