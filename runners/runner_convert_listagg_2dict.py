import argparse
from os.path import expanduser, join
import gzip
import pickle
import article_analysis.parse as aap


def run(input_path, output_path, head=-1):
    suffix = 'pgz'
    prefix = 'ngrams_corpus'
    indx = aap.get_indices(input_path, prefix, suffix)
    if head > 0:
        indx = indx[:head]

    for ii in indx[:]:
        chunk = aap.get_chunk(input_path, prefix, ii)
        chunk_new = {}
        for doi, paper in chunk.items():
            chunk_new[doi] = {}
            for o in paper.keys():
                chunk_new[doi][o] = dict(paper[o].items())
        with gzip.open(join(output_path, 'ngrams_dict_{0}.pgz'.format(ii)), 'wb') as fp:
            pickle.dump(chunk_new, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--head',
                        default=-1, type=int,
                        help='size of data batches')

    parser.add_argument('-s', '--sourcepath',
                        default='~/data/jstor/amj_raw/json_amj.txt',
                        help='path to data file')

    parser.add_argument('-d', '--destpath',
                        default='~/data/jstor/ngrams_new',
                        help='folder to write data to')

    args = parser.parse_args()
    run(args.sourcepath, args.destpath, args.head)

