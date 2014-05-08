def quick_1974((C, t), A, B):
    """
    t is duration
    C is coherence
    A is threshold coherence (given duration) for 81.6% correctness
    B is slope of psychometric curve around A

    From Kiani et al. (2008)
    """
    return 1 - 0.5*exp(-pow(C/A, B))
