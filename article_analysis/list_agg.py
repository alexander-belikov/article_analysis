def _count_elements(mapping, iterable):
    mapping_get = mapping.get
    for elem in iterable:
        mapping[elem] = mapping_get(elem, []) + elem


class ListAggregator(dict):
    def __init__(self, *args, **kwds):
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got {0}'.format(len(args)))
        super().__init__()
        self.update(*args, **kwds)

    def __missing__(self, key):
        """
        The count of elements not in the Counter is zero.

        :param key:
        :return:
        """
        return []

    def __add__(self, other):
        if not isinstance(other, ListAggregator):
            return NotImplemented
        result = ListAggregator()
        for key, item in self.items():
            newitem = item + other[key]
            if newitem:
                result[key] = newitem
        for key, item in other.items():
            if key not in self and item:
                result[key] = item
        return result

    def __iadd__(self, other):
        for elem, count in other.items():
            self[elem] += count
        return self

    def from_counter(self, counter, index):
        for key, item in counter.items():
            self[key] += item*[index]
        return self

    # def update(*args, **kwds):
    #     # The regular dict.update() operation makes no sense here because the
    #     # replace behavior results in the some of original untouched counts
    #     # being mixed-in with all of the other counts for a mismash that
    #     # doesn't have a straight-forward interpretation in most counting
    #     # contexts.  Instead, we implement straight-addition.  Both the inputs
    #     # and outputs are allowed to contain zero and negative counts.
    #
    #     if not args:
    #         raise TypeError("descriptor 'update' of 'Counter' object "
    #                         "needs an argument")
    #     self, *args = args
    #     if len(args) > 1:
    #         raise TypeError('expected at most 1 arguments, got %d' % len(args))
    #     iterable = args[0] if args else None
    #     if iterable is not None:
    #         if isinstance(iterable, Mapping):
    #             if self:
    #                 self_get = self.get
    #                 for elem, count in iterable.items():
    #                     print(elem, count)
    #                     self[elem] = count + self_get(elem, 0)
    #             else:
    #                 super(ListAggregator, self).update(iterable)
    #         else:
    #             _count_elements(self, iterable)
    #     if kwds:
    #         self.update(kwds)
