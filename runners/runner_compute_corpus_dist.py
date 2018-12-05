import argparse
from os.path import expanduser, join
import gzip
import pickle
import article_analysis.parse as aap
from article_analysis.ngram_tools import NgramAggregator


def run(input_path, output_path, head=-1, verbose=False):
    suffix = 'pgz'
    prefix = 'ngrams_dict'
    indx = aap.get_indices(input_path, prefix, suffix)
    if head > 0:
        indx = indx[:head]

    ngagg = NgramAggregator(list(range(1, 6)))
    for ii in indx:
        chunk = aap.get_chunk(input_path, prefix, ii)
        if verbose:
            print('Processing batch {0}...'.format(ii))
        ngagg.update_with_ngram_dicts(chunk.values(), verbose=verbose)
        if verbose:
            print('{0} batch processed'.format(ii))

    ngram_dist = ngagg.yield_distribution(verbose=verbose)

    with gzip.open(join(output_path, 'corpus_ngram_dist.pgz'), 'wb') as fp:
        pickle.dump(ngram_dist, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--head',
                        default=-1, type=int,
                        help='size of data batches')

    parser.add_argument('-s', '--sourcepath',
                        default=expanduser('~/data/jstor/latest'),
                        help='path to data file')

    parser.add_argument('-d', '--destpath',
                        default=expanduser('~/data/jstor/latest'),
                        help='folder to write data to')

    parser.add_argument('--verbosity',
                        default=True, type=bool,
                        help='True for verbose output')

    args = parser.parse_args()
    run(args.sourcepath, args.destpath, args.head, args.verbosity)

