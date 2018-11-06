class ListAggregator(dict):
    def __init__(*args, **kwds):
        if not args:
            raise TypeError("descriptor '__init__' of 'Counter' object "
                            "needs an argument")
        self, *args = args
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got {0}'.froamt(len(args)))
        super(ListAggregator, self).__init__()
        self.update(*args, **kwds)

    def __missing__(self, key):
        """
        The count of elements not in the Counter is zero.

        :param key:
        :return:
        """
        return []

    def __add__(self, other):
        '''
        Add counts from two.

        '''
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