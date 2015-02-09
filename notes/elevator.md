# temporal integration

## threshold and sensitivity to signal

* in the face of a noisy stimulus, you have some baseline ability to make a two-alternative decision
* the signal level (vs. noise) at which you are ~75% correct is your threshold
* => lower threshold, higher sensitivity to the signal in the stimulus

## temporal integration of signal

* you can also gain confidence if you are allowed to view the stimulus for a longer amount of time
* if we suppose that each frame of the stimulus gives us an independent signal sample, we can see what happens to sensitivity as time goes on
* threshold is related to s.e. of sampling, and should increase ~ 1/sqrt(t), where t is the # of samples, or frames
    <= if each sample is random variable X, then variance of (X+X+...+X) is nσ, where σ is s.dev of X. so s.e. of mean of sample is σ/sqrt(n)
    => so on a log-log plot of threshold vs. duration, ideal integration would have a slope of 0.5
* from Burr/Santoro: eventually, improvement in sensitivity and threshold with time will cease; improvement saturates
    => slope of 0.0 beyond some stimulus duration

