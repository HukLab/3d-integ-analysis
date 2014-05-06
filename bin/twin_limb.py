def twin_limb(x, x0, S0, p):
    """
    p is the slope at which the value increases
    x0 is the time at which the value plateaus
    S0 is the value after the value plateaus

    twin-limb function [Burr, Santoto (2001)]
        S(x) =
                {
                    S0 * (x/x0)^p | x < x0
                    S0             | x >= x0
                }
    """
    return S0 if x >= x0 else S0*pow(x/x0, p)
