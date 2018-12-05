from numpy import array
from scipy.stats import beta


class AlphabetAccumulator(object):
    def __init__(self, ):
        self.dist = dict()
        self.processed = 0
        self.lens = []

    def update_with_counts(self, counts, counts_type='list'):
        total_sum_counts = sum(self.lens)
        if counts_type == 'list':
            sum_counts = sum([len(x) for x in counts.values()])
        else:
            sum_counts = sum(counts.values())

        self.lens.append(sum_counts)

        for k, v in counts.items():
            if counts_type == 'list':
                a = len(v)
            else:
                a = v
            if k in self.dist.keys():
                self.dist[k] += array([a, sum_counts - a])
            else:
                self.dist[k] = array([0.5 + a, 0.5 + sum_counts - a + total_sum_counts])

        keys_outside_counts = set(self.dist.keys()) - set(counts.keys())

        for k in list(keys_outside_counts):
            self.dist[k] += array([0, sum_counts])

    # TODO update with AlphabetAccumulator

    def yield_distribution(self, alpha=0.05):
        intervals_dict = dict()
        for k, v in self.dist.items():
            a, b = v
            intervals_dict[k] = a/(a + b), beta.ppf(alpha, a, b), beta.ppf(1-alpha, a, b), a, b
        return intervals_dict


class NgramAggregator(object):
    def __init__(self, orders):
        self.orders = list(orders)
        self.agg = {k: AlphabetAccumulator() for k in self.orders}

    def update_with_ngram_dict(self, ngdict, counts_type='list'):
        if set(self.orders) == set(ngdict.keys()):
            for k, v in ngdict.items():
                self.agg[k].update_with_counts(v, counts_type)

        else:
            raise ValueError('in update_with_ngram_dict : ngram orders of ndict are incompatible')

    def update_with_ngram_dicts(self, ngdicts, counts_type='list'):
        for item in ngdicts:
            self.update_with_ngram_dict(item, counts_type)

    def yield_distribution(self, alpha=0.05, verbose=False):
        dist = dict()
        orders = sorted(self.agg.keys())
        for k in orders:
            dist[k] = self.agg[k].yield_distribution(alpha)
            if verbose:
                print('{0} order dist processed'.format(k))
        return dist
