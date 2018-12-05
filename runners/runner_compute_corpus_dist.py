import argparse
from os.path import expanduser, join
import gzip
import pickle
import article_analysis.parse as aap
from article_analysis.ngram_tools import NgramAggregator


def run(input_path, output_path, head=-1):
    suffix = 'pgz'
    prefix = 'ngrams_dict'
    indx = aap.get_indices(input_path, prefix, suffix)
    if head > 0:
        indx = indx[:head]

    ngagg = NgramAggregator(list(range(1, 6)))
    for ii in indx:
        chunk = aap.get_chunk(input_path, prefix, ii)
        ngagg.update_with_ngram_dicts(chunk.values())

    ngram_dist = ngagg.yield_distribution()

    with gzip.open(join(output_path, 'corpus_ngram_dist.pgz'), 'wb') as fp:
        pickle.dump(ngram_dist, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--head',
                        default=-1, type=int,
                        help='size of data batches')

    parser.add_argument('-s', '--sourcepath',
                        default=expanduser('~/data/jstor/amj_raw/json_amj.txt'),
                        help='path to data file')

    parser.add_argument('-d', '--destpath',
                        default=expanduser('~/data/jstor/ngrams_new'),
                        help='folder to write data to')

    args = parser.parse_args()
    run(args.sourcepath, args.destpath, args.head)

