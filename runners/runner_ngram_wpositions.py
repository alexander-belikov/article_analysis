import pandas as pd
import gzip
import pickle
import warnings
import argparse
from os.path import expanduser, join, isfile
import article_analysis.parse as aap
from nltk.stem.wordnet import WordNetLemmatizer

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run(input_path, output_path, head, chunksize, save=True):
    cnt_type = 'list_agg'

    fname_input = expanduser(input_path)
    print('Reading file... {0}'.format(fname_input))
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

    article_dict = {}

    if head > 0:
        corpus_split = corpus_split[:head]

    for j, chunk in zip(range(len(corpus_split)), corpus_split):
        corpus_ngrams = {}
        for doi, article in chunk.items():
            print('{0} '.format(doi), end='')
            article_phrases_ = aap.transform_article(article)
            article_dict[doi] = article_phrases_
            article_phrases = aap.lower_rm_stopwords_digits(article_phrases_)
            print('{1} '.format(doi, len(article_phrases)), end='')
            ngram_dict = aap.compute_ngrams(article_phrases, counter_type=cnt_type)
            orders = sorted(ngram_dict.keys())
            for o in orders:
                ngram_dict[o] = aap.transform_counter(ngram_dict[o], 'lemma', lmtzr, counter_type=cnt_type)
            corpus_ngrams[doi] = ngram_dict
        if save:
            with gzip.open(join(destpath, 'ngrams_corpus_{0}.pgz'.format(j + max_index)), 'wb') as fp:
                pickle.dump(corpus_ngrams, fp)
        print('\n{0} chunks, {1:.2f} % complete'.format(j, 100*(j+1)/len(corpus_split)))
    if save:
        with gzip.open(join(destpath, 'corpus_clean_dict.pgz'), 'wb') as fp:
            pickle.dump(article_dict, fp)


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

    parser.add_argument('-c', '--chunksize',
                        default='200', type=int,
                        help='chunk size of output (output list size)')

    parser.add_argument('--save',
                        default=True, type=bool,
                        help='save results')

    args = parser.parse_args()
    run(args.sourcepath, args.destpath, args.head, args.chunksize, args.save)
