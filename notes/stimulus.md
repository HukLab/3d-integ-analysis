# NOTES
    * noise dots are going to be about half signal? see line 108 in czubaDots_KTZ

==========================================

# PRESENTATION
    frate = 60 Hz
    contrast = 0.3
    depth = 0.35 monocular deg (3d only)
    gray background in aperture, 1/f noise in surround
    aperture radius = 3 deg
    center of dot stimulus = (0 deg, -3 deg)

# DOTS
    40 dots: half white, half black
    size = .15 deg
    lifetime = 250 ms
    speed = 1 deg/sec
    spacing = 0.5 deg

# TRIAL DURATIONS
    durations = (40 ms, 1200 ms)
    durations are log-spaced between minimum and maximum duration
    to create duration bins, split the durations into (100/X)% quantiles, where X is the number of duration bins
        e.g. find the edges of the 10%, 20%, ..., 100% quantiles
        this ensures that there are an equal number of trials in each duration bin

# TRIAL STRUCTURE
    2 directions
    6 coherences (usually)
    10 duration bins
    3 trials per duration per coherence
    => 360 trials per block
    inter stimulus interval = 300 ms
    sound feedback
