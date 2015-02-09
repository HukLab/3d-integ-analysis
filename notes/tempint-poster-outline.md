# Temporal integration of 2D and 3D motion discrimination

## Definition

* characterizing temporal integration consists of measuring sensitivity to some signal as a function of duration

## Motivation

* majority of motion in world is 3D
* temporal motion integration well-studied in 2D but not in 3D

## Questions

* Hypothesis: humans known to be less sensitive to 3D than to 2D - simple downward shift?

## Methods

* 2AFC psychophysics discriminating motion left/right or towards/away
* two sets of experiments, each varying motion coherence and range of durations
* only difference between 2D vs. 3D stimulus is one of screens is mirrored
* goal: measure sensitivity to some signal as a function of duration

## Results

### Data

* increasing motion coherence and duration both improve discrimination accuracy
    - surface plot

* integration saturates for both around ~1sec
    - sat-exp curves?

* psychometric curves

* subjects less sensitive to 3D than 2D
    - bloch plot

### Stages

1. Sensory stage ~100 msec
2. Decision stage ~1 sec
3. Saturation stage (following other stages)

### 2. Decision stage - "perfect integration"

* "perfect" integration has slope of 0.5
    - if subject forms decision by signal averaging independent samples of noisy stimulus
* 2D slope ~ 0.5
* 3D slope < 0.5

## Conclusions

* 3D shows lower overall sensitivity across all durations
* 3D shows sub-perfect integration in decision stage (where 2D's is near-perfect, a la Burr/Santoro)
* evidence for different mechanism of 3D motion integration

-----------

* Bloch's law: up until some critical duration, the intensity of light to get you to threshold (i.e. perceptual strength) decreases with time
    - i.e. sensitivity or SNR is linear with duration (prior to the critical duration)
    - i.e. intensity thresholds sum linearly with time
    - i.e. slope of 1.0 on log-log sensitivity-duration plot: S(t) = S_0 * t
    - only past the critical duration is the light detected perceptually--otherwise the response is guess-like (Watson 1979 via Leor's notes?)
    - same relationship is found for contrast thresholds, so contrast thresholds also sum linearly with time

* Q: Why is SNR ~ sensitivity?
    - Q: Does reaching threshold in detection tasks mean reaching a threshold in SNR?
* Q: Is V1 is a local motion detector? (e.g., MT integrates over 2+ frames, V1 only over 2 frames?)
* Q: Does the random dot stimulus has no local motion signal?
    - Q: confirm how stimulus works.

-----------

* temporal integration of 2D and 3D motion discrimination
* characterizing temporal integration consists of measuring sensitivity to some signal as a function of duration
* 2D case is well studied, and known to follow three stages:
    1. sensory stage ~100 msec
    2. decision stage ~1 sec
    3. saturation stage (following other stages)
* in the decision stage, there is defined, a priori, "perfect" integration:
    - if each sample (with same mean and variance) is summed indepently...
        - the means sum, and the variances sum
        - so given N samples, the averaged (summed) samples should have mean ~ duration and variance ~ duration
            - see: http://en.wikipedia.org/wiki/Signal_averaging
        - => stddev = sqrt(variance) = sqrt(n)*stddev, i.e. stddev ~ sqrt(n)
    - also, sensitivity = SNR = mean/stddev
        - see: http://en.wikipedia.org/wiki/Signal-to-noise_ratio_(imaging)
    - => sensitivity = SNR = n*mean / sqrt(n)*stddev = sqrt(n) * SNR
    - so sensitivity increases with the sqrt of duration (as does the stddev)
    - => 0.5 slope in log-log sensitivity-duration plot
* 3D shows lower overall sensitivity across all durations
* 3D shows sub-perfect integration in decision stage (where 2D's is near-perfect, a la Burr/Santoro)

https://github.com/mobeets/drift-diffusion
