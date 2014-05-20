LKCLAB
Leor
2012-TemporalIntegration

--------------------------------
Experiment (~) --> runDots_KTZ_data/
--------------------------------
# pre-Platypus era
runDots_KTZ.m

# stimulus length distribution: between min and max length, NOT uniform
# 	want constant "hazard function", so find pdf s.t. cdf is uniform distribution
#   => inverse exponential between min and max

--------------------------------
Analysis/ --> Figures/
--------------------------------
run_Analysis.m

--------------------------------
ISSUES
--------------------------------
* problems fitting saturating exponential # http://link.springer.com/article/10.1007%2FBF02442268
	* sensitivity to asymptote choice!

--------------------------------
EXP INFO
--------------------------------
360 trials per session
	10 duration bins
	6 coherences
	2 directions
	3 per dir per coh

	=> ~60 trials per coherence
	=> ~36 trials per duration

caveats
	different subjects get different coherences
	and different duration bins!

binEdges = [0.04 0.06 0.08 0.10 0.13 0.16 0.20 0.30 0.45 0.90 1.30 2.00]


--------------------------------
PLOTS
--------------------------------
1. % correct vs. % coherence
1a. % correct vs. % coherence, for four groups of duration

2a. % correct vs. % coherence, for each duration
2b. 75% threshold vs. duration

3a. % correct vs. % coherence, for each session index
3b. 75% threshold vs. session index

4a. % correct vs. duration, for each % coherence
4b. exp. slope vs. % coherence


--------------------------------
IDEAS
--------------------------------
* fit twin limb function vs. fit exponential [via Watamaniuk (1992); Burr, Santoro (2001)]

* plot 2b with log axes to see Block's Law?
	=> yes, but no flat line
	* Burr, Santoro (2001) showed temporal integration up to 1 sec, so this is in line with that

---------------------------------------
NOTES on sensitivity plot (1/y of 2b.)
---------------------------------------
B&S := Burr, Santoro (2001)

* all 2d sensitivity plots appear to fit "twin-limb" function, with t_0 at about 300ms
	=> according to B&S, this is evidence of Block's Law followed by end of temporal integration
* 3d plots never appear to flatline...suggesting we haven't shown long enough stimuli?
* KLB and KRM are incredibly similar in both 2d and 3d

LKC/KTZ's 3d exponential fits are _way_ affected by that last point for long durations
	=> log-log plot makes it more explained as outlier from line

---------------------------------------
NOTES from code on 2014-04-24
---------------------------------------
NO FITS:
    klb 2d: 100%
    krm 2d: 50%, 100%
    lkc 3d: 100%
    huk 3d: 0% --> fixed by run with TNC solver

* B=0.5 not always true
    * sometimes there is obviously a delay, i.e. 0.5 extends up until some duration t'
* MLE: bad at curves with lots of p~1
* MLE: shouldn't always take first guess

TO DO:
    x re-gen for ALL
    x re-gen for SUBJ
    x finer bins for ALL
    x 2d and 3d on same plot
    o fit taus with line

---------------------------------------
QUESTIONS
---------------------------------------
x Ideal observer for dot motion?

x Why black and white dots?
	=> Would multiple colors help disambiguate the correspondence problem?

* I'm guessing the 3d dots are in a cylinder? And that when they get too far forward they loop back? And that 2d has no disparity?
	=> But what if the 2d motion dots were also 3d in a cylinder? That way seeing depth is required in both (b/c disparity and motion processing interact; see: Domini et al 2003)

---------------------------------------
05/15/14
---------------------------------------

fix proportion correct and _then_ plot taus--i.e. tau vs coh-index instead of straight-up coh
    fix X% correct, blend curves above and below
    => one tau per condition per subject
play with different duration bins

Say given X, cohA has pA and cohB has pB.
    Then if you're sampling from each with probability p...call it W.
        X = B(1, pA)
        Y = B(1, pB)
        Z = B(1, p)
    We want to know W = ZX + (1-Z)Y
        W ~ B(1, p*pA + (1-p)*pB) = B(1, p(pA - pB) + pB)

        W has mean p(pA - pB) + pB and variance (p(pA - pB) + pB)*(1-(p(pA - pB) + pB))
    Thus solutions for p given pA, pB, such that W has mean P...
        P = p(pA - pB) + pB
            => p = (P-pB)/(pA-pB)

