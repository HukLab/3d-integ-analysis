n.b. make figures article-ready paper size stuff

## Figure 1a: stimulus design

* see photo on phone

## Figure 1b: pmfs without duration

* get rid of space in early x-axis (take minimum x and divide by two for xlim)
* ALL pmf: "N=5"
* 75% line
* coherence labels on/near x-axis, line as arrow

## Figure 2a: surface plots

* grid only for coherence and duration values actually present (hoping that leaves you with only those grid lines...)
* color shading to mark duration bins
* grayscale as height color, perhaps

## Figure 2b: pmfs for duration slices

* 75% line
* get rid of space in early x-axis (take minimum x and divide by two for xlim)
* remove threshold dotted lines?

## Figure 3: bloch plot

* use color code from 2b for each point
* drop the m2=0.0 slope
* move elbow point labels to ticks on x-axis
* play around with how to label slopes of m0, m1
    - text in color
    - where to place?
    - add prefix, e.g. "m1="
* make sure both elbows are visible ('--' and '-' maybe)
* DO WE TRUST THAT FIRST POINT?
    - have to treat 2d/3d the same, and have (at least) posthoc criterion for dropping them
    - e.g. any sensitivity below 2 means threshold that is extrapolated (i.e. > 50% coherence)--what does that cancel out?
    - look at pmfs for these weird durations
    - try refitting dropping the first one in each and see how fits change

## Figure 4a: sat-exp

* check plots--look at that patterned undershoot. is there delay?
* try weibull, might have nicer parameters

## Figure 4b: sat-exp

* A plot as 3d vs. 2d?
* tau plot as 3d vs. 2d?

## Figure 5?: residuals

* only fit up to first elbow point
* 5a: residuals of one-line fit
* 5b: residuals of twin-line fit
* instead of dots, do shaded area of difference from points to x-axis (i.e. 0 residuals)

## Figure 6?: histograms of slopes

* x-axis is slope, y-axis is frequency in ALL fits
* have vertical lines labeling "probability summation" and "perfect integration"
* put 2d and 3d's m0 and m1 as the four groups
