class Averager:
    def __init__(self):
        self._average = None
        self._elements_count = 0

    def add_element(self, x):
        if self._average is None:
            self._average = x
            self._elements_count = 1
        else:
            self._average = (self._average * self._elements_count + x) / (self._elements_count + 1)
            self._elements_count += 1

    @property
    def outcome(self):
        return self._average


def choose_two(lizt):
    """
    An iterator over different couples in the list

    :param lizt: A list of items
    """
    for i in range(len(lizt)):
        for j in range(len(lizt)):
            if i != j:
                yield lizt[i], lizt[j]
