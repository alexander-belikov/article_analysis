import pandas as pd
import gzip, pickle
import warnings
import argparse
from os.path import expanduser, join, isfile
import article_analysis.parse as aap
from nltk.stem.wordnet import WordNetLemmatizer

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run(input_path, output_path, head, chunksize):
    cnt_type = 'list_agg'

    fname_input = expanduser(join(input_path, 'json_amj.txt'))
    df = pd.read_json(fname_input, lines=True)

    corpus = dict(zip(df['doi'], [x['ocr'] for x in df['data']]))

    print('length of corpus: {0}'.format(len(corpus)))

    destpath = expanduser(output_path)
    suffix = 'pgz'
    prefix = 'ngrams_corpus'

    print('len corpus', len(corpus))
    ss, max_index = aap.get_present_keys(destpath, prefix, suffix)
    keys_to_process = set(corpus.keys()) - ss
    print('keys to process len', len(keys_to_process))
    remaining_corpus = {k: corpus[k] for k in keys_to_process}
    print('remaning corpus len', len(remaining_corpus))
    corpus_split = aap.split_corpus(remaining_corpus, chunksize)

    lmtzr = WordNetLemmatizer()

    if head > 0:
        corpus_split = corpus_split[:head]

    for j, chunk in zip(range(len(corpus_split)), corpus_split):
        corpus_ngrams = {}
        for doi, article in chunk.items():
            print('{0} '.format(doi), end='')
            article_phrases = aap.transform_article(article)
            print('{1} '.format(doi, len(article_phrases)), end='')
            ngram_dict = aap.compute_ngrams(article_phrases, counter_type=cnt_type)
            orders = sorted(ngram_dict.keys())
            for o in orders:
                ngram_dict[o] = aap.transform_counter(ngram_dict[o], 'lemma', lmtzr, counter_type=cnt_type)
            corpus_ngrams[doi] = ngram_dict
        with gzip.open(join(destpath, 'ngrams_corpus_{0}.pgz'.format(j + 1 + max_index )), 'wb') as fp:
            pickle.dump(corpus_ngrams, fp)
        print('\n{0} chunks, {1:.2f} % complete'.format(j, 100*j/len(corpus_split)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    parser.add_argument('-h', '--head',
                        default=-1, type=int,
                        help='size of data batches')

    parser.add_argument('-s', '--sourcepath',
                        default='~/data/jstor/amj_raw/json_amj.txt',
                        help='path to data file')

    parser.add_argument('-d', '--destpath',
                        default='~/data/jstor/ngrams_new2',
                        help='folder to write data to')

    parser.add_argument('-d', '--destpath',
                        default='~/data/jstor/ngrams_new2',
                        help='folder to write data to')

    parser.add_argument('-c', '--chunksize',
                        default='200', type=int,
                        help='chunk size of output (output list size)')

    run(args.sourcepath, args.destpath, args.head)
