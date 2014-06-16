class CommonEqualityMixin(object):
    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

class Session():
    def __init__(self, subject, dotmode, duration_bins, index):
        self.subject = subject
        self.dotmode = dotmode # 2d / 3d
        self.duration_bins = duration_bins # edges, where the first one is lower bound of whole set
        self.index = index # for given subject/dotmode; e.g. 1, 2, 3, ...

    def __eq__(self, other):
        return self.subject == other.subject and self.dotmode == other.dotmode and self.index == other.index

    def __ne__(self, other):
        return not self.__eq__(other)

class Trial(CommonEqualityMixin):
    def __init__(self, session, index, coherence, duration, duration_index, direction, response, correct):
        self.session = session
        self.index = index # within session; e.g. 1, 2, 3, ...
        self.coherence = coherence
        self.duration = duration
        self.duration_index = duration_index
        self.direction = direction # 0 / 1
        self.response = response # 1 / 2
        self.correct = correct # True / False
